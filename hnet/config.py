"""Configuration dataclasses shared by model, FLOPs calculator, trainer and eval.

Conventions
-----------
* Stage index s = 0 is the outermost (byte-level) stage; s = S is the main network.
* Per-stage fields are lists indexed by stage (len == n_stages + 1). A scalar is
  broadcast to every stage.
* `arch_layout` follows goombalab/hnet: a nested list
  ``[enc0, [enc1, [main], dec1], dec0]``. Layer codes: ``m``/``M`` = Mamba-2,
  ``t``/``T`` = attention; upper case adds a SwiGLU MLP to the block.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, List, Optional, Union

import yaml

BYTE_VOCAB = 256
BOS_ID = 254  # 0xFE never occurs in valid UTF-8 (matches goombalab/hnet ByteTokenizer)
EOS_ID = 255  # 0xFF never occurs in valid UTF-8
IGNORE_INDEX = -100


def _per_stage(v, n: int, name: str):
    if isinstance(v, (list, tuple)):
        assert len(v) == n, f"{name}: expected {n} per-stage values, got {len(v)}"
        return list(v)
    return [v] * n


@dataclass
class SSMConfig:
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 256


@dataclass
class AttnConfig:
    # Every field may be a scalar or a per-stage list.
    num_heads: Any = 16
    num_kv_heads: Any = None  # None -> same as num_heads (MHA)
    rotary_emb_dim: Any = None  # None -> full head_dim; official H-Net uses head_dim // 2
    rope_theta: Any = 10000.0
    window_size: Any = -1  # -1 = global; otherwise causal sliding window (flash-attn convention: keys in [t-w, t])
    nope_every: Any = 0  # 0 = never; k>0: layers with (layer_idx + 1) % k == 0 have no RoPE (SmolLM3 no_rope_layer)
    qkv_bias: bool = False
    out_bias: bool = False
    # Per-head width; None -> d_model // num_heads. Set explicitly (e.g. 128) for heads * head_dim != d_model
    # (Cartesia ladder: heads = depth, head_dim 128, so the attention inner width may exceed d_model).
    head_dim: Any = None


@dataclass
class DCConfig:
    """Dynamic chunking. Component fields are registry names (hnet.registry); each may be a
    scalar or a per-stage list (len == n_stages, outer stage first)."""

    # Target downsampling ratio N^s for each non-innermost stage (len == n_stages).
    target_ratios: List[float] = field(default_factory=lambda: [4.0, 2.0])
    ratio_loss_alpha: float = 0.03
    router: Any = "cosine_lookahead"  # paper / LH-Net router; see hnet/model/routing
    router_kwargs: Any = field(default_factory=dict)
    decision: Any = "threshold"  # "threshold" (p > 0.5, paper) or "quantile" (top 1/N by rank; see routing/decide.py)
    decision_kwargs: Any = None  # QuantileDecider options: {temp: 2.0, normalize: true, momentum: 0.99}
    # chunk_vocab: a hashed vocabulary for the learned chunks (model/chunk_vocab.py), dict or per-stage list:
    #   {buckets: 65536, n_hash: 2, max_len: 16, pred_weight: 0.5, n_neg: 4096}; null = off. Needs "end" routers.
    #   Optional suffix keys: suffix_lens: [2, 3, ...] (each <= max_len), suffix_gate_init: 0.0 -- the chunk vector also
    #   gets the gated rows of every listed-length string ending at the chunk's last byte (keys that depend on the
    #   chunk end only, not on the previous boundary); absent / [] = off.
    chunk_vocab: Any = None
    # router_consistency_weight: penalty on the share of the stage-0 router logit's variance that is NOT explained by
    #   the last router_consistency_n bytes (within-group / total variance, scale-free; a tokenizer is a function of the
    #   string, not of the context); see the logbook: contextual routers start out 92% consistent and training
    #   destroys it.
    router_consistency_weight: float = 0.0
    router_consistency_n: int = 4
    # router_prior_weight: tokenizer curriculum. For the first router_prior_frac of training the stage-0 router's logit
    #   is also trained (BCE, weight decaying linearly to 0) toward "a chunk ends at every whitespace byte", BPE's
    #   pre-tokenization. Steers the early race toward the word tokenizer that the adopted EXP-018/023 seeds found by
    #   themselves; after the curriculum the router is free.
    router_prior_weight: float = 0.0
    router_prior_frac: float = 0.1
    chunker: Any = "select"  # keep vectors at boundaries (paper default); "pool" = learned summary over the chunk
    chunker_kwargs: Any = field(default_factory=dict)
    dechunker: Any = "ema"  # EMA smoothing (paper default); "repeat" = no smoothing
    ste: Any = True  # multiply dechunked output by STE(selected_probs)
    aux_loss: Any = "ratio"  # paper ratio loss
    aux_loss_scope: str = "local"  # "local" = per-device then averaged (official code); "global" = all-reduced F,G
    # Hierarchy-use interventions (off by default; see logbook EXP-008 and hnet.analysis.diagnose):
    # residual_dropout: training-time probability (scalar or per stage) of zeroing a whole segment's residual path
    #   into the decoder, so the decoder must sometimes predict from the dechunked main-network output alone.
    residual_dropout: Any = 0.0
    # residual_init: "zero" (official: at init the decoder sees only the dechunked hierarchy, which is noise, and the
    #   probe run shows the router then switches the hierarchy off) or "identity" (the decoder starts from the
    #   encoder output, so the untrained hierarchy is an addition rather than the only signal).
    residual_init: str = "zero"
    # residual_warmup: fraction of training over which the residual path into each decoder is scaled from 0 to 1
    #   ("bottleneck first": early on all information must pass through the hierarchy, so the main network learns
    #   chunk-level representations before the decoder may bypass it). 0 = off.
    residual_warmup: float = 0.0
    # residual_gate: "none" or "margin": the residual path into each decoder is scaled by the stage router's mean
    #   confidence margin 2*mean|p-0.5| (detached), so the bypass only opens once the router has decided (logbook
    #   "Deep dive: the adoption race": the bypass opens at steps 60-130 while the router is still undecided).
    residual_gate: str = "none"
    # chunk_ste: multiply the chunk vectors entering the main network by STE(selected_probs), so chunk-level losses
    #   (next-chunk CE, main_aux) reach the router directly instead of only through the decoder's use of the output.
    chunk_ste: bool = False
    # router_gain_weight: a usefulness objective for the stage-0 router (training only). The stage-0 decoder + LM
    #   head are run a second time on the residual bypass alone (no gradient); the per-byte gain CE_residual_only -
    #   CE_full is credited to the chunk covering the byte, standardized against running statistics, and the router's
    #   decisions inside each chunk are trained with REINFORCE on that advantage (-A * log c_t). Measured motivation
    #   (work log): the hierarchy's gain is 2.4x larger under a word tokenizer than under a suffix tokenizer at step
    #   500, while the next-chunk CE is identical. Extra compute: one decoder + head pass per step (not in FLOPs).
    router_gain_weight: float = 0.0
    router_gain_ema: float = 0.99
    # main_aux_weight: weight of an auxiliary next-byte CE on the stage-0 dechunked main-network output alone
    #   (before the residual is added), a direct learning signal for everything below stage 0.
    main_aux_weight: float = 0.0
    # chunk_pred_k / chunk_pred_weight: chunk-level multi-byte prediction. A training-only head on the innermost
    #   main network's OUTPUT (before dechunking) predicts the first k bytes after each innermost position's byte
    #   (end semantics: the next chunk's first k bytes). A direct next-chunk training signal for the main network,
    #   bypassing the dechunker, the residual sum and the byte decoder (logbook: the main network's output carries
    #   less next-byte information than its input). 0 = off.
    chunk_pred_k: int = 0
    chunk_pred_weight: float = 0.0
    # router_entropy_weight: weight of the mean binary entropy of boundary probabilities (all stages); the ratio
    #   loss has zero gradient w.r.t. G once F is on target, so nothing else pushes decisions away from p = 0.5.
    router_entropy_weight: float = 0.0

    def for_stage(self, s: int, n_stages: int) -> dict:
        out = {}
        for name in ("router", "router_kwargs", "decision", "decision_kwargs", "chunk_vocab", "chunker", "chunker_kwargs", "dechunker", "ste",
                     "aux_loss", "residual_dropout"):
            v = getattr(self, name)
            if isinstance(v, list):
                assert len(v) == n_stages, f"dc.{name}: expected {n_stages} values"
                v = v[s]
            out[name] = v
        out["target_ratio"] = self.target_ratios[s]
        return out


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, list]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    d_intermediate: List[int] = field(default_factory=list)  # 0 = no MLP
    vocab_size: int = BYTE_VOCAB
    tie_embeddings: bool = False
    ssm: SSMConfig = field(default_factory=SSMConfig)
    attn: AttnConfig = field(default_factory=AttnConfig)
    dc: DCConfig = field(default_factory=DCConfig)
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    # Per-stage LR multipliers (outer first), "auto" (paper App. C:
    # lambda^s = sqrt(N_GPT * D^S / D^s / prod_{i<s} N^i), see hnet.optim) or None (all 1.0).
    lr_multiplier: Any = None
    lr_multiplier_n_gpt: float = 4.6  # bytes per BPE token of the training data (paper: GPT-2 on FineWeb-Edu)
    # Optimizer kinds the per-stage multipliers apply to under Muon (subset of muon / embed / head / other; see
    # hnet/optim/muon.py). None = all (default). [embed, head, other]: Muon matrices keep lr_mult 1.
    lr_multiplier_kinds: Optional[List[str]] = None
    # Layout code -> mixer registry name. Lower case = no MLP, upper case = + SwiGLU MLP. Either one dict for every
    # stage, or a per-stage list of dicts (e.g. swap only the main network's attention: [{..}, {..}, {"t": "mla"}]).
    mixer_codes: Any = field(default_factory=lambda: {"m": "mamba2", "t": "attn"})
    # Extra constructor kwargs per mixer registry name, e.g. {"mla": {"kv_lora_rank": 64}}. A value may itself be a
    # per-stage list. Merged on top of the mixer's own ``config_kwargs(cfg, stage)``.
    mixer_cfg: dict = field(default_factory=dict)
    # Block Attention Residuals (Kimi, arXiv 2603.15031) in every isotropic stack of a stage: 0 = standard residuals,
    # S > 0 = softmax attention over depth with blocks of S sub-layers (1 = Full AttnRes). Scalar or per-stage list.
    attn_res: Any = 0
    # Hashed byte n-gram embeddings added to the byte embedding (hnet.model.ngram; BLT-style): {"ns": [3, ..., 8],
    # "buckets": int or one per n}. Empty = off. Parameter-rich and FLOP-free (lookups, like the BPE embedding).
    ngram_embed: dict = field(default_factory=dict)
    # Prefix-trie output vocabulary (hnet.model.trie_head): a hashed table of chunk-prefix rows scored against the
    # main network's next-chunk prediction and added to the byte logits, {"rows": int, "dim": 32, "n_hash": 2,
    # "max_len": 16, "query": "main" | "final"}. Empty = off. Needs a 1+-stage H-Net with an "end" stage-0 router.
    trie_head: dict = field(default_factory=dict)
    # Optimizer-only (Muon), next to lr_multiplier because it is an H-Net-only LR-allocation knob: decoupled weight
    # decay of the residual bypass matrices (``residual_proj.weight``), fed to ``train.muon.residual_wd`` by
    # hnet.optim.build_optimizer. None = off (default). Spectral-norm fixed point ~1/wd; see hnet/optim/muon.py.
    residual_proj_wd: Optional[float] = None

    def attn_res_for_stage(self, s: int) -> int:
        return int(_per_stage(self.attn_res, self.n_stages + 1, "attn_res")[s])

    def mixer_codes_for_stage(self, s: int) -> dict:
        codes = self.mixer_codes
        if isinstance(codes, list):
            assert len(codes) == self.n_stages + 1, "per-stage mixer_codes needs one dict per stage"
            codes = codes[s]
        return dict(codes)

    def mixer_cfg_for_stage(self, name: str, s: int) -> dict:
        v = self.mixer_cfg.get(name, {})
        if isinstance(v, list):
            assert len(v) == self.n_stages + 1, f"per-stage mixer_cfg[{name!r}] needs one entry per stage"
            v = v[s]
        return dict(v or {})

    @property
    def n_stages(self) -> int:
        """Number of hierarchy levels below the main network (0 = isotropic)."""
        n, layout = 0, self.arch_layout
        while len(layout) == 3:
            n += 1
            layout = layout[1]
        return n

    def stage_layout(self, s: int) -> list:
        layout = self.arch_layout
        for _ in range(s):
            layout = layout[1]
        return layout

    def attn_for_stage(self, s: int) -> dict:
        n = self.n_stages + 1
        out = {}
        for f in fields(self.attn):
            out[f.name] = _per_stage(getattr(self.attn, f.name), n, f.name)[s]
        if out["num_kv_heads"] is None:
            out["num_kv_heads"] = out["num_heads"]
        explicit = out.pop("head_dim")
        head_dim = int(explicit) if explicit else self.d_model[s] // out["num_heads"]
        if explicit:  # only passed on when set, so mixers without the knob are unaffected
            out["head_dim"] = head_dim
        if out["rotary_emb_dim"] is None:
            out["rotary_emb_dim"] = head_dim
        return out

    def validate(self):
        if isinstance(self.ssm, dict):
            self.ssm = SSMConfig(**self.ssm)
        if isinstance(self.attn, dict):
            self.attn = AttnConfig(**self.attn)
        if isinstance(self.dc, dict):
            self.dc = DCConfig(**self.dc)
        n = self.n_stages + 1
        assert len(self.d_model) == n, "d_model must have one entry per stage"
        assert len(self.d_intermediate) == n, "d_intermediate must have one entry per stage"
        assert len(self.dc.target_ratios) == self.n_stages
        for s in range(1, n):
            assert self.d_model[s] >= self.d_model[s - 1], "widths must be non-decreasing"
        return self

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d: dict) -> "HNetConfig":
        d = copy.deepcopy(d)
        ssm = SSMConfig(**d.pop("ssm", {}))
        attn = AttnConfig(**d.pop("attn", {}))
        dc = DCConfig(**d.pop("dc", {}))
        return cls(ssm=ssm, attn=attn, dc=dc, **d).validate()

    @classmethod
    def from_official_json(cls, path_or_dict, target_ratios=None) -> "HNetConfig":
        """Load a goombalab/hnet config json (e.g. configs/official/hnet_2stage_XL.json)."""
        if isinstance(path_or_dict, dict):
            d = path_or_dict
        else:
            with open(path_or_dict) as f:
                d = json.load(f)
        ssm = SSMConfig(**{k: v for k, v in d.get("ssm_cfg", {}).items()})
        a = d.get("attn_cfg", {})
        attn = AttnConfig(
            num_heads=a.get("num_heads", 16),
            rotary_emb_dim=a.get("rotary_emb_dim"),
            window_size=a.get("window_size", -1),
            rope_theta=10000.0,
        )
        cfg = cls(
            arch_layout=d["arch_layout"],
            d_model=d["d_model"],
            d_intermediate=d["d_intermediate"],
            vocab_size=d.get("vocab_size", BYTE_VOCAB),
            tie_embeddings=d.get("tie_embeddings", False),
            ssm=ssm,
            attn=attn,
        )
        n_st = cfg.n_stages
        if target_ratios is None:
            # Paper: 1-stage = 6-DC, 2-stage = (3,3)-DC.
            target_ratios = {1: [6.0], 2: [3.0, 3.0]}.get(n_st, [4.0] * n_st)
        cfg.dc = DCConfig(target_ratios=list(target_ratios))
        return cfg.validate()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DenseConfig:
    """Isotropic token-level baseline. `variant` is a "dense" registry name:
    "llama" = paper's BPE Transformer (Llama/Transformer++: RMSNorm, RoPE, SwiGLU; GQA/NoPE optional),
    "modded" = modded-nanogpt-style GPT (see hnet/model/dense_modded.py)."""

    variant: str = "llama"
    vocab_size: int = 50304  # GPT-2 50257 padded to a multiple of 128
    d_model: int = 768
    n_layers: int = 12
    d_intermediate: int = 2048  # paper convention: round_up(8*D/3, 128)
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # None -> MHA
    head_dim: Optional[int] = None  # None -> d_model // num_heads; may differ (attention inner width != d_model)
    rope_theta: float = 10000.0
    nope_every: int = 0
    tie_embeddings: bool = False
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    mixer: str = "attn"  # sequence-mixer registry name for every layer (e.g. "mla", "kda")
    mixer_kwargs: dict = field(default_factory=dict)  # extra constructor kwargs for that mixer
    attn_res: int = 0  # Block Attention Residuals block size (0 = off); see HNetConfig.attn_res
    extra: dict = field(default_factory=dict)  # variant-specific knobs

    @classmethod
    def from_dict(cls, d: dict) -> "DenseConfig":
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def model_config_from_yaml(path: str):
    """A run yaml has a top-level `model:` section with `type: hnet|dense`."""
    d = load_yaml(path)["model"]
    d = dict(d)
    kind = d.pop("type", "hnet")
    if kind == "hnet":
        return HNetConfig.from_dict(d)
    if kind == "dense":
        return DenseConfig.from_dict(d)
    raise ValueError(kind)


def apply_overrides(d: dict, overrides) -> dict:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) to a nested dict in place."""
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        assert _, f"override must look like key=value, got {ov!r}"
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        val = yaml.safe_load(raw)
        if isinstance(val, str):  # PyYAML (YAML 1.1) leaves e.g. "3e-4" as a string
            try:
                val = float(val)
            except ValueError:
                pass
        node[parts[-1]] = val
    return d
