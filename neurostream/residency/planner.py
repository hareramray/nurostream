"""Residency planning: decide what lives where, once, at load time.

Placement is a knapsack over benefit density. For a tensor of size B accessed
F times per token, keeping it in a tier of bandwidth W saves roughly

    benefit = F * B * (1/W_disk - 1/W_tier)   seconds per token

and it costs B bytes of that tier. Sorting by benefit-per-byte and filling
greedily is the standard fractional-knapsack move; it is not optimal for the
0/1 case but the error is bounded by one tensor and the inputs are estimates
anyway.

The interesting result on this hardware is that the ordering is not "biggest
first" or "smallest first". Attention projections and norms are touched every
token and are small, so they dominate. MoE experts are touched perhaps 5% of
tokens each and are large, so they lose — which is exactly why a 235B model
is tractable and a 70B dense one is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..format.gguf import GGMLType

# Measured on the target machine (GB/s). See BUILD_PROMPT.md.
BW_VRAM = 350.0
BW_RAM = 18.2
BW_DISK = 2.35
# Approximate per-request cost; otherwise tiny, always-used norms lose to
# large matrices despite saving hundreds of small reads on every token.
READ_LATENCY = 80e-6


@dataclass
class Placement:
    vram: list[str] = field(default_factory=list)
    ram: list[str] = field(default_factory=list)
    disk: list[str] = field(default_factory=list)
    vram_bytes: int = 0
    ram_bytes: int = 0
    disk_bytes: int = 0

    def summary(self) -> str:
        return (
            f"VRAM {self.vram_bytes / 1e9:5.2f}GB ({len(self.vram)} tensors) | "
            f"RAM {self.ram_bytes / 1e9:5.2f}GB ({len(self.ram)}) | "
            f"disk {self.disk_bytes / 1e9:6.2f}GB ({len(self.disk)})"
        )

    def est_bytes_per_token(self) -> float:
        return (
            self.vram_bytes / BW_VRAM
            + self.ram_bytes / BW_RAM
            + self.disk_bytes / BW_DISK
        )


def access_frequency(name: str, cfg) -> float:
    """Expected reads per token, in [0, 1].

    Everything outside an expert is read every token. An expert is read only
    when the router picks it.
    """
    if ".ffn_" in name and "_exps" in name:
        if cfg is not None and getattr(cfg, "n_expert", 0) > 0:
            return cfg.n_expert_used / cfg.n_expert
        return 0.1
    return 1.0


def is_expert_tensor(name: str) -> bool:
    return ".ffn_" in name and "_exps" in name and name.endswith('.weight')


def plan(
    gguf,
    cfg=None,
    vram_budget: int = 0,
    ram_budget: int = 0,
    reserve_vram: int = 0,
    slab_budget: int = 0,
    float_itemsize: int = 2,
) -> Placement:
    """Greedy benefit-density placement over all tensors in the file.

    Expert tensors are excluded from whole-tensor VRAM placement when a slab
    budget is set: only 8 of 128 experts fire per token, so residency is far
    better spent on the hot slabs (chosen from measured routing) than on all
    128 experts of whichever layers happen to fit.
    """
    vram_budget = max(0, vram_budget - reserve_vram - slab_budget)

    scored = []
    for name, info in gguf.tensors.items():
        if slab_budget and is_expert_tensor(name):
            continue
        f = access_frequency(name, cfg)
        if name == "token_embd.weight" and "output.weight" in gguf.tensors:
            # Decode gathers one row. A tied embedding is also the output
            # projection, however, and still needs the entire matrix.
            f = 1 / max(1, info.torch_shape[0])
        b = info.nbytes
        vram_cost = (
            info.n_elements * float_itemsize
            if info.dtype in (GGMLType.F32, GGMLType.F16, GGMLType.BF16)
            else b
        )
        # Compute density directly to avoid floating-point multiply/divide
        # noise arbitrarily reordering equally hot tensors.
        latency_density = READ_LATENCY / b if b <= 64 * 1024 else 0.0
        gain_vram = f * ((1 / BW_DISK - 1 / BW_VRAM) / 1e9 + latency_density)
        gain_ram = f * ((1 / BW_DISK - 1 / BW_RAM) / 1e9 + latency_density)
        scored.append((gain_vram * (b / vram_cost), gain_ram, name, b, vram_cost))

    # Benefit density is size-independent for equally-hot tensors, so the sort
    # would otherwise be arbitrary and whichever tensors happened to land last
    # would spill. Break ties by size so the largest hot tensors win VRAM.

    # VRAM first, by density
    p = Placement()
    taken: set[str] = set()
    for _, _, name, b, vram_cost in sorted(scored, key=lambda r: (-r[0], -r[3])):
        # VRAM keeps quantized tensors quantized — see cache.py. Only tensors
        # already stored as float cost more than their on-disk size.
        if p.vram_bytes + vram_cost <= vram_budget:
            p.vram.append(name)
            p.vram_bytes += vram_cost
            taken.add(name)

    # RAM next, still quantized, so cost is the on-disk size
    for _, _, name, b, _cost in sorted(scored, key=lambda r: (-r[1], -r[3])):
        if name in taken:
            continue
        if p.ram_bytes + b <= ram_budget:
            p.ram.append(name)
            p.ram_bytes += b
            taken.add(name)

    for name, info in gguf.tensors.items():
        if name not in taken:
            p.disk.append(name)
            p.disk_bytes += info.nbytes
    return p


def apply_plan(cache, placement: Placement, verbose: bool = False) -> None:
    """Materialise a placement into a TieredCache."""
    for i, name in enumerate(placement.vram):
        if not cache.admit_vram(name):
            placement.ram.append(name)
        if verbose and i % 50 == 0:
            print(f"    vram {i}/{len(placement.vram)}", flush=True)
    for i, name in enumerate(placement.ram):
        cache.admit_ram(name)
        if verbose and i % 50 == 0:
            print(f"    ram  {i}/{len(placement.ram)}", flush=True)
    cache.pin(placement.vram)
