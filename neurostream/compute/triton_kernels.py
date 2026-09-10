"""Fused dequantize-and-multiply kernels in Triton.

The reason these exist: expanding a quantized tensor to fp16 and then calling
torch.matmul moves the *dequantized* weights through VRAM. For the 8B model's
LM head that is 1.24 GB of traffic and ~6 s of kernel time per token, which
puts a hard floor of well under 1 tok/s on the whole engine no matter how
good the I/O path is.

A fused kernel never materialises the dequantized weights at all. It reads the
4.5-bit blocks, expands them in registers, multiplies, and accumulates. The
weights cross the memory bus once, compressed. That is the same trick the
streaming engine plays against the SSD, applied one tier further up.

Triton rather than CUDA C++ because this machine has no MSVC and no nvcc, and
Triton ships its own compiler.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    # fp16 scale fields are read through a separate fp16 view of the same
    # buffer rather than reassembled from bytes: Triton promotes the shift in
    # `(hi << 8) | lo` to int32, which makes a bitcast to fp16 meaningless.
    # Every block stride here is even, so the halved offset is always exact.

    @triton.jit
    def q4k_gemv_kernel(
        x_ptr,          # (K,) fp32
        w_ptr,          # (N, row_bytes) uint8, Q4_K blocks
        h_ptr,          # same buffer viewed as fp16
        y_ptr,          # (N,) fp32
        N, K, row_bytes,
        BLOCK_N: tl.constexpr,
    ):
        """y[n] = sum_k x[k] * dequant(W)[n, k], one block of rows per program."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        row_base = rows * row_bytes

        # Element layout inside a 256-value Q4_K block.
        j = tl.arange(0, 256)
        s = j // 32          # sub-block 0..7
        l = j % 32
        g = s // 2
        half = s % 2
        qs_off = 16 + g * 32 + l

        # Scale/min byte picks, per get_scale_min_k4.
        s_lt4 = s < 4
        a_off = 4 + tl.where(s_lt4, s, s - 4)
        b_off = 4 + tl.where(s_lt4, s + 4, s)
        c_off = 4 + s + 4

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        nblk = K // 256

        for kb in range(nblk):
            blk = row_base[:, None] + kb * 144

            hbase = (row_base + kb * 144) // 2
            d = tl.load(h_ptr + hbase, mask=rmask, other=0.0).to(tl.float32)[:, None]
            dmin = tl.load(h_ptr + hbase + 1, mask=rmask, other=0.0).to(tl.float32)[:, None]

            m2 = rmask[:, None]
            a = tl.load(w_ptr + blk + a_off[None, :], mask=m2, other=0).to(tl.int32)
            b = tl.load(w_ptr + blk + b_off[None, :], mask=m2, other=0).to(tl.int32)
            c = tl.load(w_ptr + blk + c_off[None, :], mask=m2, other=0).to(tl.int32)

            lt4 = s_lt4[None, :]
            sc = tl.where(lt4, a & 63, (c & 0x0F) | ((a >> 6) << 4)).to(tl.float32)
            mn = tl.where(lt4, b & 63, (c >> 4) | ((b >> 6) << 4)).to(tl.float32)

            qb = tl.load(w_ptr + blk + qs_off[None, :], mask=m2, other=0).to(tl.int32)
            q = ((qb >> (4 * half[None, :])) & 0x0F).to(tl.float32)

            w = d * sc * q - dmin * mn

            xv = tl.load(
                x_ptr + kb * 256 + j, mask=(kb * 256 + j) < K, other=0.0
            ).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)

        tl.store(y_ptr + rows, acc, mask=rmask)

    @triton.jit
    def q6k_gemv_kernel(
        x_ptr, w_ptr, h_ptr, y_ptr,
        N, K, row_bytes,
        BLOCK_N: tl.constexpr,
    ):
        """Q6_K variant: ql[128] | qh[64] | scales[16] int8 | d fp16."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        row_base = rows * row_bytes

        j = tl.arange(0, 256)
        halfsel = j // 128           # which 128-element half
        r = j % 128
        lane = r // 32               # 0..3 -> q1..q4
        l = r % 32
        ql_off = halfsel * 64 + tl.where(lane % 2 == 0, l, l + 32)
        qh_off = 128 + halfsel * 32 + l
        shift = lane * 2
        sc_off = 192 + halfsel * 8 + (l // 16) + lane * 2

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        nblk = K // 256

        for kb in range(nblk):
            blk = row_base[:, None] + kb * 210
            d = tl.load(
                h_ptr + (row_base + kb * 210 + 208) // 2, mask=rmask, other=0.0
            ).to(tl.float32)[:, None]
            m2 = rmask[:, None]

            ql = tl.load(w_ptr + blk + ql_off[None, :], mask=m2, other=0).to(tl.int32)
            qh = tl.load(w_ptr + blk + qh_off[None, :], mask=m2, other=0).to(tl.int32)
            sc = tl.load(w_ptr + blk + sc_off[None, :], mask=m2, other=0).to(tl.int32)
            sc = ((sc + 128) % 256 - 128).to(tl.float32)  # int8

            nib = tl.where(lane[None, :] < 2, ql & 0x0F, ql >> 4)
            q = (nib | (((qh >> shift[None, :]) & 3) << 4)) - 32
            w = d * sc * q.to(tl.float32)

            xv = tl.load(
                x_ptr + kb * 256 + j, mask=(kb * 256 + j) < K, other=0.0
            ).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)

        tl.store(y_ptr + rows, acc, mask=rmask)


if HAVE_TRITON:
    @triton.jit
    def quant32_gemv_kernel(x_ptr, w_ptr, h_ptr, y_ptr, N, K, row_bytes,
                            BLOCK_N: tl.constexpr, MX: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        j = tl.arange(0, 256)
        acc = tl.full((BLOCK_N,), 0, tl.float32)
        for base in range(tl.cdiv(K, 256)):
            col = base * 256 + j
            mask = (rows[:, None] < N) & (col[None, :] < K)
            if MX:
                off = rows[:, None] * row_bytes + (col[None, :] // 32) * 17
                e = tl.load(w_ptr + off, mask=mask, other=0).to(tl.int32)
                bits = tl.where(e < 2, 0x00200000 << e, (e - 1) << 23)
                scale = bits.to(tl.float32, bitcast=True)
                packed = tl.load(w_ptr + off + 1 + col[None, :] % 16, mask=mask, other=0).to(tl.int32)
                code = (packed >> tl.where(col[None, :] % 32 < 16, 0, 4)) & 15
                mag = code & 7
                value = tl.where(mag < 4, mag, tl.where(mag == 4, 4,
                        tl.where(mag == 5, 6, tl.where(mag == 6, 8, 12)))).to(tl.float32)
                weight = scale * tl.where(code < 8, value, -value)
            else:
                off = rows[:, None] * row_bytes + (col[None, :] // 32) * 34
                scale = tl.load(h_ptr + off // 2, mask=mask, other=0).to(tl.float32)
                q = tl.load(w_ptr + off + 2 + col[None, :] % 32, mask=mask, other=0).to(tl.int32)
                weight = scale * tl.where(q < 128, q, q - 256).to(tl.float32)
            xv = tl.load(x_ptr + col, mask=col < K, other=0).to(tl.float32)
            acc += tl.sum(weight * xv[None, :], axis=1)
        tl.store(y_ptr + rows, acc, mask=rows < N)


_KERNELS = {}
if HAVE_TRITON:
    from ..format.gguf import GGMLType

    _KERNELS = {
        GGMLType.Q4_K: (q4k_gemv_kernel, 144),
        GGMLType.Q6_K: (q6k_gemv_kernel, 210),
        GGMLType.Q8_0: (quant32_gemv_kernel, 34),
        GGMLType.MXFP4: (quant32_gemv_kernel, 17),
    }


def can_fuse(dtype, x: torch.Tensor) -> bool:
    """Fused path applies to single-vector decode on CUDA."""
    return (
        HAVE_TRITON
        and x.is_cuda
        and x.shape[0] == 1
        and dtype in _KERNELS
    )


def fused_gemv(
    raw: torch.Tensor,
    dtype,
    x: torch.Tensor,
    n_rows: int,
    K: int,
    out_dtype: torch.dtype = torch.float16,
    block_n: int = 16,
    num_warps: int = 2,
) -> torch.Tensor:
    """y = x @ dequant(raw).T without materialising the weights.

    raw: (n_rows * row_bytes,) uint8 on CUDA. x: (1, K).
    """
    kernel, block_bytes = _KERNELS[dtype]
    # Row stride, not block size. Each row spans K/256 quantization blocks.
    from ..format.gguf import TYPE_LAYOUT
    row_bytes = (K // TYPE_LAYOUT[dtype][0]) * block_bytes
    # x stays fp16; the kernel widens after load. Converting here would
    # mean an extra alloc + kernel launch on every one of ~250 calls a token.
    xf = x.reshape(-1).contiguous()
    y = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    hview = raw if dtype == GGMLType.MXFP4 else raw.view(torch.float16)
    grid = (triton.cdiv(n_rows, block_n),)
    extra = {'MX': dtype == GGMLType.MXFP4} if dtype in (GGMLType.MXFP4, GGMLType.Q8_0) else {}
    kernel[grid](
        xf, raw, hview, y, n_rows, K, row_bytes,
        BLOCK_N=block_n, num_warps=num_warps, **extra,
    )
    return y.reshape(1, n_rows).to(out_dtype)
