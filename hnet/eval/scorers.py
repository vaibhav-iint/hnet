"""Log-likelihood scorers with one interface for byte models, our BPE models and HF checkpoints.

    scorer.loglikelihood(requests: list[tuple[str, str]]) -> list[tuple[float, bool, int]]
        # (sum logprob of the continuation in nats, greedy?, n_bytes of the scored continuation)
    scorer.loglikelihood_detailed(requests) -> list[LLResult]   # + n_units (tokens or bytes)
    scorer.rolling_loglikelihood(texts) -> list[tuple[float, int]]  # whole-text logprob, UTF-8 bytes

Prompt boundary handling (identical for every scorer, lm-eval/lighteval convention)
------------------------------------------------------------------------------------
Trailing whitespace of the context is moved to the front of the continuation
(``"Answer: " + "x"`` -> ``"Answer:" + " x"``), so the scored text always carries its leading space.
``n_bytes`` is the UTF-8 length of the continuation *after* that move.

* Byte models: sequence = ``[BOS_ID] + utf8(context) + utf8(continuation)``; the continuation bytes are scored.
* Token models (tiktoken/HF): lm-eval ``_encode_pair``: ``whole = enc(context + continuation)``,
  ``ctx = enc(context)``, continuation tokens = ``whole[len(ctx):]``, model input = ``ctx + cont``. If the
  continuation merges completely into the last context token (``len(ctx) == len(whole)``) we follow lighteval and
  score the merged last token. An empty context is replaced by a single prefix token (lm-eval: EOS).
* Prefix/BOS for token models: ``TokenModelScorer`` (our dense models, trained on documents that start with
  ``<|endoftext|>``) prepends EOT by default, mirroring the byte model's BOS. ``HFScorer`` follows lm-eval: it
  prepends BOS only if the tokenizer does so by default (GPT-2/SmolLM: no), unless ``add_bos`` is given.

Long inputs are left-truncated to ``max_len`` model positions (the continuation is never truncated).
Requests are length-sorted and either packed into one ``cu_seqlens`` batch (our models, SPEC forward API) or
right-padded (HF models, causal so padding never leaks into scored positions).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from hnet.config import BOS_ID

Request = Tuple[str, str]


@dataclass
class LLResult:
    logprob: float  # sum over scored continuation units (nats)
    greedy: bool  # every continuation unit is the argmax
    n_bytes: int  # UTF-8 bytes of the (space-adjusted) continuation
    n_units: int  # scored positions: tokens (token models) or bytes (byte models)

    def as_tuple(self):
        return (self.logprob, self.greedy, self.n_bytes)


def split_trailing_space(context: str, continuation: str) -> Tuple[str, str]:
    """Move trailing whitespace of the context into the continuation (lm-eval/lighteval)."""
    n = len(context) - len(context.rstrip())
    if n > 0:
        continuation = context[-n:] + continuation
        context = context[:-n]
    return context, continuation


def _default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BaseScorer:
    """Shared batching / log-prob gathering. Subclasses implement ``encode_pair`` and ``_logits``."""

    unit = "token"
    name = "model"

    def __init__(self, max_len: int = 2048, batch_tokens: int = 16384, device=None, packed: bool = True,
                 amp_dtype: Optional[torch.dtype] = None):
        self.max_len = int(max_len)
        self.batch_tokens = int(batch_tokens)
        self.device = torch.device(device) if device is not None else _default_device()
        self.packed = packed
        if amp_dtype is None and self.device.type == "cuda":
            amp_dtype = torch.bfloat16
        self.amp_dtype = amp_dtype

    # ------------------------------------------------------------------ to implement
    def encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int], int]:
        """-> (context ids incl. any prefix, continuation ids, n_bytes of the continuation)."""
        raise NotImplementedError

    def encode_text(self, text: str) -> Tuple[List[int], List[int]]:
        """-> (prefix ids, text ids) for rolling / bpb scoring."""
        raise NotImplementedError

    def _logits(self, inputs: List[List[int]], ncont: List[int]) -> List[torch.Tensor]:
        """Run the model on a batch of input sequences; for each, return the logits of its last ``n`` positions
        ([n, V], the rows that predict the continuation)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ public API
    def loglikelihood(self, requests: Sequence[Request]) -> List[Tuple[float, bool, int]]:
        return [r.as_tuple() for r in self.loglikelihood_detailed(requests)]

    def loglikelihood_detailed(self, requests: Sequence[Request], progress: bool = False) -> List[LLResult]:
        seqs, ncont, nbytes = [], [], []
        for ctx, cont in requests:
            c_ids, x_ids, nb = self.encode_pair(ctx, cont)
            assert len(c_ids) >= 1 and len(x_ids) >= 1, (ctx, cont)
            seqs.append(c_ids + x_ids)
            ncont.append(len(x_ids))
            nbytes.append(nb)
        scores = self.score_sequences(seqs, ncont, progress=progress)
        return [LLResult(lp, g, nb, n) for (lp, g), nb, n in zip(scores, nbytes, ncont)]

    def rolling_loglikelihood(self, texts: Sequence[str]) -> List[Tuple[float, int]]:
        """Total log-likelihood of each text (lm-eval ``loglikelihood_rolling`` with disjoint windows of
        ``max_len`` targets; each window is conditioned on the unit just before it, the first one on the prefix).
        Returns (sum logprob, UTF-8 bytes of the text)."""
        seqs, ncont, owner = [], [], []
        for i, t in enumerate(texts):
            pre, ids = self.encode_text(t)
            units = pre + ids
            start = len(pre)
            while start < len(units):
                end = min(start + self.max_len, len(units))
                seqs.append(units[start - 1:end])
                ncont.append(end - start)
                owner.append(i)
                start = end
        scores = self.score_sequences(seqs, ncont)
        tot = [0.0] * len(texts)
        for (lp, _), i in zip(scores, owner):
            tot[i] += lp
        return [(tot[i], len(t.encode("utf-8"))) for i, t in enumerate(texts)]

    @torch.no_grad()  # not inference_mode: compiled flex attention falls back to the O(T^2) eager path under it
    def score_sequences(self, seqs: List[List[int]], ncont: List[int], progress: bool = False):
        """Sum log p(seq[-n:]) given the prefix, for each (seq, n). Returns [(logprob, greedy)] in input order."""
        items = []
        for s, n in zip(seqs, ncont):
            assert 1 <= n <= self.max_len, f"continuation of {n} units exceeds max_len={self.max_len}"
            s = s[-(self.max_len + 1):]
            items.append((s[:-1], s[-n:]))
        order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
        out: List[Optional[Tuple[float, bool]]] = [None] * len(items)
        batches, cur, cur_cost = [], [], 0
        for i in order:
            L = len(items[i][0])
            # packed: total positions; padded: rows * longest row (= first, sorted descending)
            cost = cur_cost + L if self.packed else (len(cur) + 1) * (len(items[cur[0]][0]) if cur else L)
            if cur and cost > self.batch_tokens:
                batches.append(cur)
                cur, cost = [], L
            cur.append(i)
            cur_cost = cost
        if cur:
            batches.append(cur)
        it = batches
        if progress:
            try:
                from tqdm import tqdm

                it = tqdm(batches, desc=f"score[{self.name}]", leave=False)
            except ImportError:
                pass
        amp = (torch.autocast(self.device.type, dtype=self.amp_dtype)
               if self.amp_dtype is not None else nullcontext())
        for b in it:
            with amp:
                logits = self._logits([items[i][0] for i in b], [len(items[i][1]) for i in b])
            for i, lg in zip(b, logits):
                inp, tgt = items[i]
                rows = lg.float()
                lp = F.log_softmax(rows, dim=-1)
                t = torch.tensor(tgt, device=lp.device, dtype=torch.long)
                tok_lp = lp.gather(-1, t[:, None]).squeeze(-1)
                greedy = bool((rows.argmax(-1) == t).all())
                out[i] = (float(tok_lp.double().sum()), greedy)
        return out


# ====================================================================== our models (SPEC forward API)
class _OwnModelScorer(BaseScorer):
    """Calls ``model(input_ids, cu_seqlens=..., max_seqlen=..., return_logits=True).logits`` (SPEC Model API)."""

    def __init__(self, model, **kw):
        super().__init__(**kw)
        self.model = model.to(self.device).eval() if hasattr(model, "to") else model

    def _logits(self, inputs, ncont):
        dev = self.device
        if self.packed:
            lens = [len(x) for x in inputs]
            ids = torch.tensor([t for x in inputs for t in x], dtype=torch.long, device=dev)
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
            cu[1:] = torch.cumsum(torch.tensor(lens), 0)
            out = self.model(ids, cu_seqlens=cu.to(dev), max_seqlen=max(lens), return_logits=True)
            lg = out.logits.reshape(-1, out.logits.shape[-1])
            return [lg[int(cu[k + 1]) - n:int(cu[k + 1])] for k, n in enumerate(ncont)]
        # padded [B, L]: right padding, each row one segment (causal => padding never affects scored rows)
        L = max(len(x) for x in inputs)
        ids = torch.zeros(len(inputs), L, dtype=torch.long)
        for r, x in enumerate(inputs):
            ids[r, :len(x)] = torch.tensor(x)
        out = self.model(ids.to(dev), return_logits=True)
        return [out.logits[r, len(x) - n:len(x)] for r, (x, n) in enumerate(zip(inputs, ncont))]


class ByteModelScorer(_OwnModelScorer):
    """Byte-level LM (our H-Net, official H-Net, byte baselines): scores ``BOS + context + continuation``."""

    unit = "byte"
    name = "byte"

    # batch_tokens = the training micro-batch (16 x 2048 bytes): packed eval sequences never exceed what training ran
    def __init__(self, model, bos_id: int = BOS_ID, max_len: int = 8192, batch_tokens: int = 16384, **kw):
        super().__init__(model, max_len=max_len, batch_tokens=batch_tokens, **kw)
        self.bos_id = bos_id

    def encode_pair(self, context, continuation):
        context, continuation = split_trailing_space(context, continuation)
        cb = list(continuation.encode("utf-8"))
        return [self.bos_id] + list(context.encode("utf-8")), cb, len(cb)

    def encode_text(self, text):
        return [self.bos_id], list(text.encode("utf-8"))


class _TokenizerMixin:
    """lm-eval/lighteval context/continuation split for subword tokenizers."""

    prefix_id: int
    add_bos: bool
    bos_id: Optional[int]

    def _enc(self, text: str) -> List[int]:
        raise NotImplementedError

    def encode_pair(self, context, continuation):
        context, continuation = split_trailing_space(context, continuation)
        nb = len(continuation.encode("utf-8"))
        pre = [self.bos_id] if self.add_bos else []
        if context == "":
            return (pre or [self.prefix_id]), self._enc(continuation), nb
        whole = self._enc(context + continuation)
        ctx = self._enc(context)
        if len(ctx) >= len(whole):  # continuation merged into the last context token (lighteval)
            ctx = whole[:len(whole) - 1]
        cont = whole[len(ctx):]
        if not pre and not ctx:
            pre = [self.prefix_id]
        return pre + ctx, cont, nb

    def encode_text(self, text):
        pre = [self.bos_id] if self.add_bos else [self.prefix_id]
        return pre, self._enc(text)


def get_tiktoken(name: str = "gpt2"):
    import tiktoken

    return tiktoken.get_encoding(name)


class TokenModelScorer(_TokenizerMixin, _OwnModelScorer):
    """Our dense BPE models (tiktoken GPT-2 by default). ``add_bos=True`` prepends ``<|endoftext|>`` (as in
    training, where every document starts with it); set ``add_bos=False`` for exact lm-eval GPT-2 behaviour."""

    unit = "token"
    name = "token"

    def __init__(self, model, tokenizer="gpt2", add_bos: bool = True, max_len: int = 2048,
                 batch_tokens: int = 16384, **kw):
        super().__init__(model, max_len=max_len, batch_tokens=batch_tokens, **kw)
        self.tok = get_tiktoken(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.prefix_id = self.tok.eot_token
        self.bos_id = self.tok.eot_token
        self.add_bos = add_bos

    def _enc(self, text):
        return self.tok.encode_ordinary(text)


# ====================================================================== HF transformers checkpoints
class HFScorer(_TokenizerMixin, BaseScorer):
    """``transformers`` causal LM by hub name, e.g. ``gpt2``, ``HuggingFaceTB/SmolLM2-135M``,
    ``HuggingFaceTB/SmolLM3-3B-checkpoints@<revision>``. bf16 on CUDA, fp32 on CPU."""

    def __init__(self, name: str, revision: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 add_bos: Optional[bool] = None, max_len: Optional[int] = None, batch_tokens: int = 16384,
                 device=None, model=None, tokenizer=None, fast_head: Optional[bool] = None, **kw):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(device) if device is not None else _default_device()
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.tok = tokenizer or AutoTokenizer.from_pretrained(name, revision=revision)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
        self.model = model.to(device).eval()
        if fast_head is None:  # base_model + output embeddings == full forward for plain decoder LMs
            cfg = self.model.config
            fast_head = cfg.model_type in ("gpt2", "llama", "smollm3", "mistral", "qwen2", "gpt_neox") and not \
                getattr(cfg, "final_logit_softcapping", None)
        self.fast_head = bool(fast_head)
        if max_len is None:
            cfg = self.model.config
            max_len = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None) or 2048
            max_len = min(int(max_len), 4096)
        super().__init__(max_len=max_len, batch_tokens=batch_tokens, device=device, packed=False,
                         amp_dtype=None, **kw)
        self.name = name + (f"@{revision}" if revision else "")
        self.bos_id = self.tok.bos_token_id
        if add_bos is None:  # lm-eval: tokenizer default add_special_tokens
            a = self.tok.encode("a")
            b = self.tok.encode("a", add_special_tokens=False)
            add_bos = len(a) > len(b) and self.bos_id is not None and a[0] == self.bos_id
        self.add_bos = bool(add_bos)
        eos = self.tok.eos_token_id
        self.prefix_id = self.bos_id if (self.add_bos or eos is None) else eos

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def _logits(self, inputs, ncont):
        L = max(len(x) for x in inputs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(inputs), L), pad, dtype=torch.long)
        mask = torch.zeros(len(inputs), L, dtype=torch.long)
        for r, x in enumerate(inputs):
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        head = self.model.get_output_embeddings() if self.fast_head else None
        if head is not None:
            # LM head only on the scored rows (full [B, L, V] logits would dominate memory for 50k+ vocabularies)
            h = self.model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            return [head(h[r, len(x) - n:len(x)]) for r, (x, n) in enumerate(zip(inputs, ncont))]
        out = self.model(input_ids=ids, attention_mask=mask)
        return [out.logits[r, len(x) - n:len(x)] for r, (x, n) in enumerate(zip(inputs, ncont))]


def make_scorer(model, kind: str, **kw) -> BaseScorer:
    """kind: "byte" | "token"."""
    if kind == "byte":
        return ByteModelScorer(model, **kw)
    if kind == "token":
        return TokenModelScorer(model, **kw)
    raise ValueError(kind)
