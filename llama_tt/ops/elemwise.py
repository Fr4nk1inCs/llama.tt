from typing import no_type_check

import triton
import triton.language as tl

from llama_tt.dtype import Scalar
from llama_tt.tensor import DeviceTensor
from llama_tt.utils import cdiv

BLOCK_SIZE = 1024


@triton.jit
@no_type_check
def _fill_kernel(
    ptr: tl.pointer_type,
    val,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    tl.store(ptr + offsets, val, mask=mask)


def fill(tensor: DeviceTensor, value: Scalar):
    grid = (cdiv(tensor.numel(), BLOCK_SIZE),)
    _fill_kernel[grid](
        tensor,
        value,
        tensor.numel(),
        BLOCK_SIZE,
    )


@triton.jit
@no_type_check
def _add_kernel(
    a_ptr: tl.pointer_type,
    b_ptr: tl.pointer_type,
    c_ptr: tl.pointer_type,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = a_vals + b_vals
    tl.store(c_ptr + offsets, c_vals, mask=mask)


def add(
    a: DeviceTensor, b: DeviceTensor, out: DeviceTensor | None = None
) -> DeviceTensor:
    if out is None:
        out = DeviceTensor.alloc(a.shape, a.dtype, a.strides)
    grid = (cdiv(a.numel(), BLOCK_SIZE),)
    _add_kernel[grid](
        a,
        b,
        out,
        a.numel(),
        BLOCK_SIZE,
    )
    return out
