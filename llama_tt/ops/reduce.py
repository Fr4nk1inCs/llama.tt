from typing import cast, no_type_check

import triton
import triton.language as tl

from llama_tt.dtype import Scalar, bool_
from llama_tt.tensor import DeviceTensor
from llama_tt.utils import cdiv


@triton.jit
@no_type_check
def _logical_and(a, b):
    return a and b


@triton.jit
@no_type_check
def _all_eq_kernel_stage_1(
    a_ptr: tl.pointer_type,
    partial: tl.pointer_type,
    val,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    tl.assume(partial.dtype.element_ty == tl.int1)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=True)
    cmp = a_vals == val
    result = tl.reduce(cmp, None, _logical_and)
    tl.store(partial + pid, result)


@triton.jit
@no_type_check
def _all_eq_kernel_stage_2(
    partial_ptr: tl.pointer_type,
    result_ptr: tl.pointer_type,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    tl.assume(partial_ptr.dtype.element_ty == tl.int1)
    result = True
    for i in tl.range(0, numel, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE) + i
        val = tl.load(partial_ptr + offsets, mask=offsets < numel, other=True)
        result = result and tl.reduce(val, None, _logical_and)
    tl.store(result_ptr, result)


def all_eq(tensor: DeviceTensor, value: Scalar) -> bool:
    BLOCK_SIZE = 1024
    result = DeviceTensor.alloc([1], bool_)

    if tensor.numel() < BLOCK_SIZE:
        grid = (1,)
        _all_eq_kernel_stage_1[grid](
            tensor,
            result,
            value,
            tensor.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cast(bool, result.item())

    num_partials = cdiv(tensor.numel(), BLOCK_SIZE)
    partials = DeviceTensor.alloc([num_partials], bool_)
    grid_stage_1 = (num_partials,)
    _all_eq_kernel_stage_1[grid_stage_1](
        tensor,
        partials,
        value,
        tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    grid_stage_2 = (1,)
    _all_eq_kernel_stage_2[grid_stage_2](
        partials,
        result,
        num_partials,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return cast(bool, result.item())
