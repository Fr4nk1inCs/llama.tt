from typing import no_type_check

import triton.language as tl
from triton import jit


@jit
@no_type_check
def fill_kernel(
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


@jit
@no_type_check
def add_kernel(
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
