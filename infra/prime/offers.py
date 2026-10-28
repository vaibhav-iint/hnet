"""GPU offers for pods (live availability API) and sandboxes (docs price formula + overrides)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import config as C


@dataclass
class Offer:
    kind: str                 # "pod" | "sandbox"
    gpu_type: str
    gpu_count: int
    price_hr: float | None    # total $/h for the offer (incl. default disk if billed extra)
    per_gpu_hr: float | None
    provider: str = ""
    location: str = ""
    stock: str = ""
    vcpus: int | None = None
    mem_gb: int | None = None
    gpu_mem_gb: int | None = None
    note: str = ""
    raw: object = field(default=None, repr=False)


def _norm(s: str) -> str:
    return s.upper().replace(" ", "_").replace("-", "_")


def matches(gpu_type: str, query: str | None) -> bool:
    return query is None or _norm(query) in _norm(gpu_type)


def disk_extra_hr(avail) -> float:
    d = avail.disk
    if d is None or d.price_per_unit is None or d.default_included_in_price:
        return 0.0
    return (d.default_count or 0) * d.price_per_unit


def pod_offers(gpu: str | None = None, client=None, include_cpu: bool = False) -> list[Offer]:
    if client is None:
        from prime_cli.api.availability import AvailabilityClient
        from prime_cli.core import APIClient
        client = AvailabilityClient(APIClient())
    data = client.get()
    out = []
    for gtype, items in data.items():
        if gtype == "CPU_NODE" and not include_cpu:
            continue
        if not matches(gtype, gpu):
            continue
        for a in items:
            base = a.prices.community_price if a.prices.community_price is not None else a.prices.on_demand
            if base is None:
                continue
            tot = base + disk_extra_hr(a)
            n = max(1, a.gpu_count)
            out.append(Offer(
                kind="pod", gpu_type=gtype, gpu_count=a.gpu_count, price_hr=tot, per_gpu_hr=tot / n,
                provider=a.provider or "", location=a.country or a.data_center or "",
                stock=a.stock_status, vcpus=a.vcpu.default_count, mem_gb=a.memory.default_count,
                gpu_mem_gb=a.gpu_memory, note="spot" if a.is_spot else "", raw=a))
    out.sort(key=lambda o: (o.per_gpu_hr if o.per_gpu_hr is not None else 1e9, o.gpu_type))
    return out


def sandbox_gpu_prices() -> dict[str, float]:
    env = os.environ.get("HNET_PRIME_SANDBOX_GPU_PRICES")
    if env:
        return {_norm(k): float(v) for k, v in json.loads(env).items()}
    return {}


def sandbox_offers(gpu: str | None = None, pods: list[Offer] | None = None) -> list[Offer]:
    """Sandbox GPU prices are not published via the API. We show an override price if set,
    else the cheapest pod price for the same GPU type as a reference, else unknown."""
    overrides = sandbox_gpu_prices()
    pod_ref = {}
    for o in pods or []:
        if o.per_gpu_hr is not None:
            pod_ref[_norm(o.gpu_type)] = min(pod_ref.get(_norm(o.gpu_type), 1e9), o.per_gpu_hr)
    out = []
    types = list(dict.fromkeys(C.SANDBOX_GPU_TYPES + list(overrides)))
    for t in types:
        if not matches(t, gpu):
            continue
        if _norm(t) in overrides:
            p, note = overrides[_norm(t)], "price from HNET_PRIME_SANDBOX_GPU_PRICES"
        elif _norm(t) in pod_ref:
            p, note = pod_ref[_norm(t)], "price unknown; pod price shown; GPU sandboxes need a grant"
        else:
            p, note = None, "price unknown; GPU sandboxes need a grant"
        out.append(Offer(kind="sandbox", gpu_type=t, gpu_count=1, price_hr=p, per_gpu_hr=p, note=note))
    if gpu is None or matches("CPU", gpu):
        from .budget import sandbox_price_hr
        p = sandbox_price_hr(1, 2, 10)
        out.append(Offer(kind="sandbox", gpu_type="CPU (1 vCPU, 2 GiB, 10 GiB)", gpu_count=0,
                         price_hr=p, per_gpu_hr=None,
                         note="$0.02/vCPU-h + $0.0125/GiB-h + $0.0002/GiB-disk-h"))
    return out


def cheapest_pod_offer(gpu: str, count: int, offers: list[Offer]) -> Offer | None:
    cands = [o for o in offers if o.kind == "pod" and o.gpu_count == count
             and matches(o.gpu_type, gpu) and o.stock.lower() not in ("unavailable", "out of stock")]
    return min(cands, key=lambda o: o.price_hr) if cands else None


def format_table(offers: list[Offer]) -> str:
    rows = [("kind", "gpu_type", "n", "$/GPU-h", "$/h", "vcpu", "memGB", "provider", "loc", "stock", "note")]
    for o in offers:
        rows.append((o.kind, o.gpu_type, str(o.gpu_count),
                     f"{o.per_gpu_hr:.3f}" if o.per_gpu_hr is not None else "-",
                     f"{o.price_hr:.3f}" if o.price_hr is not None else "-",
                     str(o.vcpus or ""), str(o.mem_gb or ""), o.provider, o.location, o.stock, o.note))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r[:-1])) + "  " + r[-1] for r in rows)
