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


@dataclass
class DCConfig:
    """Dynamic chunking. Component fields are registry names (hnet.registry); each may be a
    scalar or a per-stage list (len == n_stages, outer stage first)."""

    # Target downsampling ratio N^s for each non-innermost stage (len == n_stages).
    target_ratios: List[float] = field(default_factory=lambda: [4.0, 2.0])
    ratio_loss_alpha: float = 0.03
    router: Any = "cosine_lookahead"  # paper / LH-Net router; see hnet/model/routing
    router_kwargs: Any = field(default_factory=dict)
    chunker: Any = "select"  # keep vectors at boundaries (paper default)
    dechunker: Any = "ema"  # EMA smoothing (paper default); "repeat" = no smoothing
    ste: Any = True  # multiply dechunked output by STE(selected_probs)
    aux_loss: Any = "ratio"  # paper ratio loss
    aux_loss_scope: str = "local"  # "local" = per-device then averaged (official code); "global" = all-reduced F,G

    def for_stage(self, s: int, n_stages: int) -> dict:
        out = {}
        for name in ("router", "router_kwargs", "chunker", "dechunker", "ste", "aux_loss"):
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
    # Layout code -> mixer registry name. Lower case = no MLP, upper case = + SwiGLU MLP. Either one dict for every
    # stage, or a per-stage list of dicts (e.g. swap only the main network's attention: [{..}, {..}, {"t": "mla"}]).
    mixer_codes: Any = field(default_factory=lambda: {"m": "mamba2", "t": "attn"})
    # Extra constructor kwargs per mixer registry name, e.g. {"mla": {"kv_lora_rank": 64}}. A value may itself be a
    # per-stage list. Merged on top of the mixer's own ``config_kwargs(cfg, stage)``.
    mixer_cfg: dict = field(default_factory=dict)

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
        head_dim = self.d_model[s] // out["num_heads"]
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
    rope_theta: float = 10000.0
    nope_every: int = 0
    tie_embeddings: bool = False
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    mixer: str = "attn"  # sequence-mixer registry name for every layer (e.g. "mla", "kda")
    mixer_kwargs: dict = field(default_factory=dict)  # extra constructor kwargs for that mixer
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
