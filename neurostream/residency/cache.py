"""Tiered weight residency: VRAM > RAM > disk.

Three storage policies, not two, because the obvious design is wrong.

Storing *dequantized* fp16 in VRAM is what you reach for first — no per-use
cost, ready to multiply. But a Q4 model is 4.5 bits per weight on disk and 16
bits dequantized, so that policy needs 3.5x the VRAM the file occupies. An 8B
Q4 model is 5 GB on disk and would need 16 GB of VRAM as fp16. It does not
fit in 8 GB; quantized, it does, with room to spare.

So the tiers are:

  VRAM-Q   raw quantized bytes on the GPU, expanded per access. The default.
           Dequant is a GPU kernel over data already in VRAM — no PCIe, no
           host involvement.
  VRAM-F   dequantized, for tensors stored as F32/F16 anyway (norms, biases),
           where "dequantizing" is just a cast and caching it costs nothing.
  RAM      raw quantized bytes on the host. 4x more weights than fp16 would
           fit, at the cost of a PCIe transfer plus dequant per access.

Eviction is LRU within each tier, with explicit pinning for always-hot
tensors the planner has marked.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from ..compute.quant import dequantize
from ..format.gguf import GGMLType

# Types that are already float — caching them expanded costs nothing extra.
_FLOAT_TYPES = {GGMLType.F32, GGMLType.F16, GGMLType.BF16}


@dataclass
class CacheStats:
    vram_hits: int = 0
    ram_hits: int = 0
    misses: int = 0
    vram_bytes: int = 0
    ram_bytes: int = 0
    evictions: int = 0
    dequant_calls: int = 0
    slab_hits: int = 0
    slab_misses: int = 0

    @property
    def slab_rate(self) -> float:
        t = self.slab_hits + self.slab_misses
        return 0.0 if t == 0 else self.slab_hits / t

    @property
    def hit_rate(self) -> float:
        total = self.vram_hits + self.ram_hits + self.misses
        return 0.0 if total == 0 else (self.vram_hits + self.ram_hits) / total

    def summary(self) -> str:
        return (
            f"vram {self.vram_bytes / 1e9:.2f}GB/{self.vram_hits} hits | "
            f"ram {self.ram_bytes / 1e9:.2f}GB/{self.ram_hits} hits | "
            f"miss {self.misses} | evict {self.evictions} | "
            f"hit-rate {self.hit_rate:.1%}"
        )


@dataclass
class _Entry:
    """Either a ready float tensor, or quantized bytes plus how to expand."""

    payload: torch.Tensor
    nbytes: int
    ready: bool
    dtype: GGMLType | None = None
    n_elements: int = 0
    shape: tuple[int, ...] = ()


class TieredCache:
    def __init__(
        self,
        source,
        device: torch.device | str = "cpu",
        compute_dtype: torch.dtype = torch.float32,
        vram_budget: int = 0,
        ram_budget: int = 0,
    ) -> None:
        self.source = source
        self.gguf = source.gguf
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.vram_budget = vram_budget
        self.ram_budget = ram_budget

        # Expert slabs are cached per (tensor, row-range). Caching a whole
        # 128-expert tensor to serve the 8 that fire is a 16x waste of VRAM;
        # caching the hot slabs across every layer is what the measured
        # routing skew actually rewards.
        self._slab: OrderedDict[str, _Entry] = OrderedDict()
        # Row spans per tensor, so a request for part of a slab can find the
        # slab that encloses it instead of only an exact-key match.
        self._slab_spans: dict[str, list[tuple[int, int]]] = {}
        self.slab_bytes = 0
        self.slab_budget = 0
        self._vram: OrderedDict[str, _Entry] = OrderedDict()
        self._ram: OrderedDict[str, _Entry] = OrderedDict()
        self._pinned: set[str] = set()
        self.stats = CacheStats()

    # -- sizing -----------------------------------------------------------

    def vram_cost(self, name: str) -> int:
        """Bytes this tensor would occupy in VRAM under the chosen policy."""
        info = self.gguf.tensors[name]
        if info.dtype in _FLOAT_TYPES:
            return info.n_elements * self.compute_dtype.itemsize
        return info.nbytes  # stays quantized

    # -- placement --------------------------------------------------------

    def pin(self, names) -> None:
        self._pinned.update(names)

    def _evict(self, store, need: int, budget: int, is_vram: bool) -> None:
        used = self.stats.vram_bytes if is_vram else self.stats.ram_bytes
        while used + need > budget and store:
            for name in list(store):
                if name in self._pinned:
                    continue
                e = store.pop(name)
                if is_vram:
                    self.stats.vram_bytes -= e.nbytes
                else:
                    self.stats.ram_bytes -= e.nbytes
                self.stats.evictions += 1
                used -= e.nbytes
                break
            else:
                return  # everything left is pinned

    def admit_vram(self, name: str) -> bool:
        if name in self._vram:
            return True
        info = self.gguf.tensors[name]
        need = self.vram_cost(name)
        if need > self.vram_budget:
            return False
        self._evict(self._vram, need, self.vram_budget, True)
        if self.stats.vram_bytes + need > self.vram_budget:
            return False

        if info.dtype in _FLOAT_TYPES:
            t = self.source.fetch(
                name, device=self.device, dtype=self.compute_dtype
            )
            entry = _Entry(payload=t, nbytes=need, ready=True)
        else:
            raw = self.source.raw_bytes(name).to(self.device)
            entry = _Entry(
                payload=raw, nbytes=need, ready=False, dtype=info.dtype,
                n_elements=info.n_elements, shape=info.torch_shape,
            )
        self._vram[name] = entry
        self.stats.vram_bytes += need
        return True

    def admit_ram(self, name: str) -> bool:
        if name in self._ram:
            return True
        info = self.gguf.tensors[name]
        need = info.nbytes
        if need > self.ram_budget:
            return False
        self._evict(self._ram, need, self.ram_budget, False)
        if self.stats.ram_bytes + need > self.ram_budget:
            return False
        self._ram[name] = _Entry(
            payload=self.source.raw_bytes(name), nbytes=need, ready=False,
            dtype=info.dtype, n_elements=info.n_elements,
            shape=info.torch_shape,
        )
        self.stats.ram_bytes += need
        return True

    # -- access -----------------------------------------------------------

    def is_resident(self, name: str) -> bool:
        return name in self._vram or name in self._ram

    def _expand(self, e: _Entry) -> torch.Tensor:
        if e.ready:
            return e.payload
        self.stats.dequant_calls += 1
        raw = e.payload
        if raw.device != self.device:
            raw = raw.to(self.device, non_blocking=True)
        flat = dequantize(
            raw, e.dtype, e.n_elements, out_dtype=self.compute_dtype
        )
        return flat.reshape(e.shape)

    def get(self, name: str) -> torch.Tensor:
        e = self._vram.get(name)
        if e is not None:
            self._vram.move_to_end(name)
            self.stats.vram_hits += 1
            return self._expand(e)

        e = self._ram.get(name)
        if e is not None:
            self._ram.move_to_end(name)
            self.stats.ram_hits += 1
            return self._expand(e)

        self.stats.misses += 1
        return self.source.fetch(
            name, device=self.device, dtype=self.compute_dtype
        )

    def _slab_for(self, name: str, start: int, end: int):
        """(entry, slab_start) for a cached slab covering rows [start, end).

        Neuron-block streaming asks for sub-ranges of the very slabs the
        planner pinned, so matching only exact keys would miss every one of
        them and re-read the block from disk — the opposite of what pinning
        is for. Spans per tensor are few (one per pinned expert), so the scan
        is cheaper than the read it avoids.
        """
        for lo, hi in self._slab_spans.get(name, ()):
            if lo <= start and end <= hi:
                key = f"{name}#{lo}:{hi}"
                entry = self._slab.get(key)
                if entry is not None:
                    self._slab.move_to_end(key)
                    return entry, lo
        return None

    def fetch_rows(
        self, name: str, start: int, end: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Row range from whichever tier holds this tensor.

        Block streaming has to work over every tier, not just disk. A resident
        quantized tensor is still 4.5 bits per weight and still has to be
        expanded; expanding all of it at once is what blows up on a 151936-row
        embedding table. Slicing rows out of resident bytes keeps the peak
        bounded no matter where the bytes live.
        """
        device = self.device if device is None else device
        dtype = self.compute_dtype if dtype is None else dtype

        from ..io.source import row_geometry

        found = self._slab_for(name, start, end)
        if found is not None:
            slab, base = found
            _, row_elems, row_bytes = row_geometry(self.gguf.tensors[name])
            self.stats.vram_hits += 1
            self.stats.slab_hits += 1
            self.stats.dequant_calls += 1
            raw = slab.payload[
                (start - base) * row_bytes : (end - base) * row_bytes
            ]
            return dequantize(
                raw.to(device), slab.dtype,
                (end - start) * row_elems, out_dtype=dtype,
            ).reshape(end - start, row_elems)

        e = self._vram.get(name) or self._ram.get(name)
        if e is None:
            self.stats.misses += 1
            return self.source.fetch_rows(
                name, start, end, device=device, dtype=dtype
            )

        if name in self._vram:
            self._vram.move_to_end(name)
            self.stats.vram_hits += 1
        else:
            self._ram.move_to_end(name)
            self.stats.ram_hits += 1

        info = self.gguf.tensors[name]
        n_rows, row_elems, row_bytes = row_geometry(info)
        start, end = max(0, start), min(n_rows, end)
        if start >= end:
            return torch.empty((0, row_elems), dtype=dtype, device=device)

        if e.ready:
            return (
                e.payload.reshape(-1, row_elems)[start:end].to(device, dtype)
            )

        raw = e.payload[start * row_bytes : end * row_bytes]
        if raw.device != torch.device(device):
            raw = raw.to(device, non_blocking=True)
        self.stats.dequant_calls += 1
        flat = dequantize(
            raw, e.dtype, (end - start) * row_elems, out_dtype=dtype
        )
        return flat.reshape(end - start, row_elems)

    def raw_quantized(self, name: str):
        """(raw_bytes_on_device, ggml_dtype) if resident and quantized.

        Covers both resident tiers. A RAM-resident tensor is uploaded still
        quantized — 38 MB of Q6_K across PCIe costs ~3 ms, where expanding it
        host-side and sending fp16 costs an order of magnitude more. Skipping
        this for the RAM tier is what left 5 spilled ffn_down tensors eating
        62% of every decode step.
        """
        e = self._vram.get(name)
        if e is not None:
            if e.ready:
                return None
            self._vram.move_to_end(name)
            self.stats.vram_hits += 1
            return e.payload, e.dtype

        e = self._ram.get(name)
        if e is None or e.ready:
            return None
        self._ram.move_to_end(name)
        self.stats.ram_hits += 1
        return e.payload.to(self.device, non_blocking=True), e.dtype

    def admit_slab(self, name: str, start: int, end: int) -> bool:
        """Pin one expert's row range into VRAM, still quantized."""
        from ..io.source import row_geometry

        info = self.gguf.tensors[name]
        if info.dtype in _FLOAT_TYPES or self.is_resident(name):
            return False
        key = f"{name}#{start}:{end}"
        if key in self._slab:
            return True
        _, _, row_bytes = row_geometry(info)
        need = (end - start) * row_bytes
        if self.slab_bytes + need > self.slab_budget:
            return False
        raw = self.source.raw_rows(name, start, end).to(self.device)
        self._slab[key] = _Entry(
            payload=raw, nbytes=need, ready=False, dtype=info.dtype,
        )
        self._slab_spans.setdefault(name, []).append((start, end))
        self.slab_bytes += need
        return True

    def has_slab(self, name: str, start: int, end: int) -> bool:
        return self._slab_for(name, start, end) is not None

    def raw_rows(self, name: str, start: int, end: int):
        """(raw_bytes_on_device, ggml_dtype) for a row range, or None.

        The fused-kernel counterpart to fetch_rows. Used for MoE experts,
        where a "row range" is one expert's slab.
        """
        from ..io.source import row_geometry

        info = self.gguf.tensors[name]
        if info.dtype in _FLOAT_TYPES:
            return None
        _, _, row_bytes = row_geometry(info)
        found = self._slab_for(name, start, end)
        if found is not None:
            slab, base = found
            self.stats.vram_hits += 1
            self.stats.slab_hits += 1
            return (
                slab.payload[
                    (start - base) * row_bytes : (end - base) * row_bytes
                ],
                slab.dtype,
            )
        if self._slab:
            self.stats.slab_misses += 1

        lo, hi = start * row_bytes, end * row_bytes

        e = self._vram.get(name)
        if e is not None and not e.ready:
            self._vram.move_to_end(name)
            self.stats.vram_hits += 1
            return e.payload[lo:hi], e.dtype
        e = self._ram.get(name)
        if e is not None and not e.ready:
            self._ram.move_to_end(name)
            self.stats.ram_hits += 1
            return e.payload[lo:hi].to(self.device, non_blocking=True), e.dtype

        self.stats.misses += 1
        raw = self.source.raw_rows(name, start, end)
        return raw.to(self.device, non_blocking=True), info.dtype

    def submit_rows(self, name: str, start: int, end: int):
        """Async row read, only meaningful for tensors still on disk."""
        if self.is_resident(name):
            return None
        return self.source.submit_rows(name, start, end)

    def clear(self) -> None:
        """Drop every resident tensor and hand the VRAM back.

        Without the empty_cache() the allocator keeps the arena reserved, so
        a second model loaded in the same process silently competes with the
        first one's corpse.
        """
        self._slab.clear()
        self._slab_spans.clear()
        self._vram.clear()
        self._ram.clear()
        self.slab_bytes = 0
        self.stats = CacheStats()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
