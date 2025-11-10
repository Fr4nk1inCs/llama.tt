from collections.abc import Callable
from typing import Any, TypeVar, no_type_check

import triton.language as tl
from triton import JITFunction, jit

from llama_tt.kernels.utils import (broadcast_shapes, compute_pid4d,
                                    make_three_block_ptr)
from llama_tt.lighter.tensor.base import TensorBase
from llama_tt.utils import cdiv


@jit
@no_type_check
def _binary1d_template(
    in0_ptr: tl.pointer_type,
    in1_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    numel,
    BLOCK_SIZE: tl.constexpr,
    FN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    in0 = tl.load(in0_ptr + offs, mask=mask)
    in1 = tl.load(in1_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, FN(in0, in1), mask=mask)


@jit
@no_type_check
def _binary2d_template(
    in0_ptr: tl.pointer_type,
    in1_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    M,
    N,
    stride_0m,
    stride_0n,
    stride_1m,
    stride_1n,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N)
    block_shape = tl.constexpr((BLOCK_M, BLOCK_N))
    load_order = tl.constexpr((0, 1))
    in0_block_ptr, in1_block_ptr, out_block_ptr = make_three_block_ptr(
        in0_ptr,
        in1_ptr,
        out_ptr,
        (M, N),
        (stride_0m, stride_0n),
        (stride_1m, stride_1n),
        (stride_om, stride_on),
        block_offsets,
        block_shape,
        load_order,
    )

    in0 = tl.load(in0_block_ptr)
    in1 = tl.load(in1_block_ptr)
    tl.store(out_block_ptr, FN(in0, in1))


@jit
@no_type_check
def _binary3d_template(
    in0_ptr: tl.pointer_type,
    in1_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    M,
    N,
    K,
    stride_0m,
    stride_0n,
    stride_0k,
    stride_1m,
    stride_1n,
    stride_1k,
    stride_om,
    stride_on,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    block_offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N, pid_k * BLOCK_K)
    block_shape = tl.constexpr((BLOCK_M, BLOCK_N, BLOCK_K))
    load_order = tl.constexpr((0, 1, 2))
    in0_block_ptr, in1_block_ptr, out_block_ptr = make_three_block_ptr(
        in0_ptr,
        in1_ptr,
        out_ptr,
        (M, N, K),
        (stride_0m, stride_0n, stride_0k),
        (stride_1m, stride_1n, stride_1k),
        (stride_om, stride_on, stride_ok),
        block_offsets,
        block_shape,
        load_order,
    )

    in0 = tl.load(in0_block_ptr)
    in1 = tl.load(in1_block_ptr)
    tl.store(out_block_ptr, FN(in0, in1))


@jit
@no_type_check
def _binary4d_template(
    in0_ptr: tl.pointer_type,
    in1_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    D0,
    D1,
    D2,
    D3,
    stride_0d0,
    stride_0d1,
    stride_0d2,
    stride_0d3,
    stride_1d0,
    stride_1d1,
    stride_1d2,
    stride_1d3,
    stride_od0,
    stride_od1,
    stride_od2,
    stride_od3,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    FN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_d0 = tl.cdiv(D0, BLOCK_D0)
    num_pid_d1 = tl.cdiv(D1, BLOCK_D1)
    num_pid_d2 = tl.cdiv(D2, BLOCK_D2)
    num_pid_d3 = tl.cdiv(D3, BLOCK_D3)

    pid_d0, pid_d1, pid_d2, pid_d3 = compute_pid4d(
        pid, num_pid_d0, num_pid_d1, num_pid_d2, num_pid_d3
    )

    block_offsets = (
        pid_d0 * BLOCK_D0,
        pid_d1 * BLOCK_D1,
        pid_d2 * BLOCK_D2,
        pid_d3 * BLOCK_D3,
    )
    block_shape = tl.constexpr((BLOCK_D0, BLOCK_D1, BLOCK_D2, BLOCK_D3))
    load_order = tl.constexpr((0, 1, 2, 3))
    in0_block_ptr, in1_block_ptr, out_block_ptr = make_three_block_ptr(
        in0_ptr,
        in1_ptr,
        out_ptr,
        (D0, D1, D2, D3),
        (stride_0d0, stride_0d1, stride_0d2, stride_0d3),
        (stride_1d0, stride_1d1, stride_1d2, stride_1d3),
        (stride_od0, stride_od1, stride_od2, stride_od3),
        block_offsets,
        block_shape,
        load_order,
    )

    in0 = tl.load(in0_block_ptr)
    in1 = tl.load(in1_block_ptr)
    tl.store(out_block_ptr, FN(in0, in1))


_BLOCK_SIZE_1D = 4096
_BLOCK_SIZE_2D = 64
_BLOCK_SIZE_3D = 16
_BLOCK_SIZE_4D = 8


_T = TypeVar("_T", bound=TensorBase)


def _generate_op(
    fn: JITFunction[Callable[..., Any]],
) -> Callable[[_T, _T], _T]:
    def binary_op(in0: _T, in1: _T) -> _T:
        shape, stride0, stride1 = broadcast_shapes(
            in0.shape, in1.shape, in0.strides, in1.strides
        )
        out = in0.new_empty(*shape)
        ndim = len(shape)
        if ndim == 1:
            numel = shape[0]
            grid = (cdiv(numel, _BLOCK_SIZE_1D),)
            _binary1d_template[grid](
                in0,
                in1,
                out,
                numel,
                BLOCK_SIZE=_BLOCK_SIZE_1D,
                FN=fn,
            )
        elif ndim == 2:
            m, n = shape
            grid = (cdiv(m, _BLOCK_SIZE_2D), cdiv(n, _BLOCK_SIZE_2D))
            _binary2d_template[grid](
                in0,
                in1,
                out,
                m,
                n,
                *stride0,
                *stride1,
                *out.strides,
                BLOCK_M=_BLOCK_SIZE_2D,
                BLOCK_N=_BLOCK_SIZE_2D,
                FN=fn,
            )
        elif ndim == 3:
            m, n, k = shape
            grid = (
                cdiv(m, _BLOCK_SIZE_3D),
                cdiv(n, _BLOCK_SIZE_3D),
                cdiv(k, _BLOCK_SIZE_3D),
            )
            _binary3d_template[grid](
                in0,
                in1,
                out,
                m,
                n,
                k,
                *stride0,
                *stride1,
                *out.strides,
                BLOCK_M=_BLOCK_SIZE_3D,
                BLOCK_N=_BLOCK_SIZE_3D,
                BLOCK_K=_BLOCK_SIZE_3D,
                FN=fn,
            )
        elif ndim == 4:
            d0, d1, d2, d3 = shape
            grid = (
                cdiv(d0, _BLOCK_SIZE_4D)
                * cdiv(d1, _BLOCK_SIZE_4D)
                * cdiv(d2, _BLOCK_SIZE_4D)
                * cdiv(d3, _BLOCK_SIZE_4D),
            )
            _binary4d_template[grid](
                in0,
                in1,
                out,
                d0,
                d1,
                d2,
                d3,
                *stride0,
                *stride1,
                *out.strides,
                BLOCK_D0=_BLOCK_SIZE_4D,
                BLOCK_D1=_BLOCK_SIZE_4D,
                BLOCK_D2=_BLOCK_SIZE_4D,
                BLOCK_D3=_BLOCK_SIZE_4D,
                FN=fn,
            )
        else:
            raise NotImplementedError(
                f"binary_op not implemented for tensors with {ndim} dimensions"
            )

        return out

    return binary_op


@jit
@no_type_check
def _add(in0, in1):
    return in0 + in1


@jit
@no_type_check
def _sub(in0, in1):
    return in0 - in1


@jit
@no_type_check
def _mul(in0, in1):
    return in0 * in1


@jit
@no_type_check
def _div(in0, in1):
    return in0 / in1


add = _generate_op(_add)
sub = _generate_op(_sub)
mul = _generate_op(_mul)
div = _generate_op(_div)
