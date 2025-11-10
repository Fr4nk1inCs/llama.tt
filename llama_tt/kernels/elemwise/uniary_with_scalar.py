from collections.abc import Callable
from typing import Any, TypeVar, no_type_check

import triton.language as tl
from triton import JITFunction, jit

from llama_tt.kernels.utils import compute_pid4d, make_two_block_ptr
from llama_tt.lighter.dtype import Scalar
from llama_tt.lighter.tensor.base import TensorBase
from llama_tt.utils import cdiv


@jit
@no_type_check
def _uniary_with_scalar_1d_template(
    in_ptr: tl.pointer_type,
    scalar,
    out_ptr: tl.pointer_type,
    numel,
    BLOCK_SIZE: tl.constexpr,
    FN: tl.constexpr,
    INPLACE: tl.constexpr = False,
):
    if INPLACE:
        tl.assume(in_ptr == out_ptr)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    in_ptrs = in_ptr + offs
    out_ptrs = out_ptr + offs

    in_ = tl.load(in_ptrs, mask=mask)
    tl.store(out_ptrs, FN(in_, scalar), mask=mask)


@jit
@no_type_check
def _uniary_with_scalar_2d_template(
    in_ptr: tl.pointer_type,
    scalar,
    out_ptr: tl.pointer_type,
    M,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FN: tl.constexpr,
    INPLACE: tl.constexpr = False,
):
    if INPLACE:
        tl.assume(in_ptr == out_ptr)
        tl.assume(stride_im == stride_om)
        tl.assume(stride_in == stride_on)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N)
    block_shape = tl.constexpr((BLOCK_M, BLOCK_N))
    load_order = tl.constexpr((0, 1))
    in_block_ptr, out_block_ptr = make_two_block_ptr(
        in_ptr,
        out_ptr,
        shape=(M, N),
        strides1=(stride_im, stride_in),
        strides2=(stride_om, stride_on),
        offsets=block_offsets,
        block_shape=block_shape,
        order=load_order,
    )

    in_ = tl.load(in_block_ptr)
    tl.store(out_block_ptr, FN(in_, scalar))


@jit
@no_type_check
def _uniary_with_scalar_3d_template(
    in_ptr: tl.pointer_type,
    scalar,
    out_ptr: tl.pointer_type,
    M,
    N,
    K,
    stride_im,
    stride_in,
    stride_ik,
    stride_om,
    stride_on,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FN: tl.constexpr,
    INPLACE: tl.constexpr = False,
):
    if INPLACE:
        tl.assume(in_ptr == out_ptr)
        tl.assume(stride_im == stride_om)
        tl.assume(stride_in == stride_on)
        tl.assume(stride_ik == stride_ok)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    block_offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N, pid_k * BLOCK_K)
    block_shape = tl.constexpr((BLOCK_M, BLOCK_N, BLOCK_K))
    load_order = tl.constexpr((0, 1, 2))
    in_block_ptr, out_block_ptr = make_two_block_ptr(
        in_ptr,
        out_ptr,
        shape=(M, N, K),
        strides1=(stride_im, stride_in, stride_ik),
        strides2=(stride_om, stride_on, stride_ok),
        offsets=block_offsets,
        block_shape=block_shape,
        order=load_order,
    )

    in_ = tl.load(in_block_ptr)
    tl.store(out_block_ptr, FN(in_, scalar))


@jit
@no_type_check
def _uniary_with_scalar_4d_template(
    in_ptr: tl.pointer_type,
    scalar,
    out_ptr: tl.pointer_type,
    D0,
    D1,
    D2,
    D3,
    stride_i0,
    stride_i1,
    stride_i2,
    stride_i3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    FN: tl.constexpr,
    INPLACE: tl.constexpr = False,
):
    if INPLACE:
        tl.assume(in_ptr == out_ptr)
        tl.assume(stride_i0 == stride_o0)
        tl.assume(stride_i1 == stride_o1)
        tl.assume(stride_i2 == stride_o2)
        tl.assume(stride_i3 == stride_o3)

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
    in_block_ptr, out_block_ptr = make_two_block_ptr(
        in_ptr,
        out_ptr,
        shape=(D0, D1, D2, D3),
        strides1=(stride_i0, stride_i1, stride_i2, stride_i3),
        strides2=(stride_o0, stride_o1, stride_o2, stride_o3),
        offsets=block_offsets,
        block_shape=block_shape,
        order=load_order,
    )

    in_ = tl.load(in_block_ptr)
    tl.store(out_block_ptr, FN(in_, scalar))


_BLOCK_SIZE_1D = 4096
_BLOCK_SIZE_2D = 64
_BLOCK_SIZE_3D = 16
_BLOCK_SIZE_4D = 8

_T = TypeVar("_T", bound=TensorBase)


def _generate_op(
    fn: JITFunction[Callable[..., Any]], inplace: bool = False
) -> Callable[[_T, Scalar], _T]:
    def uniary_op_outplace(input: _T, scalar: Scalar) -> _T:
        output = input.empty_like()

        if input.dim == 1 or input.is_contiguous:
            numel = input.numel
            grid_size = (cdiv(numel, _BLOCK_SIZE_1D),)
            _uniary_with_scalar_1d_template[grid_size](
                in_ptr=input,
                scalar=scalar,
                out_ptr=output,
                numel=numel,
                BLOCK_SIZE=_BLOCK_SIZE_1D,
                FN=fn,
            )
        elif input.dim == 2:
            m, n = input.shape

            grid_size = (
                cdiv(m, _BLOCK_SIZE_2D),
                cdiv(n, _BLOCK_SIZE_2D),
            )
            _uniary_with_scalar_2d_template[grid_size](
                input,
                scalar,
                output,
                m,
                n,
                *input.strides,
                *output.strides,
                BLOCK_M=_BLOCK_SIZE_2D,
                BLOCK_N=_BLOCK_SIZE_2D,
                FN=fn,
            )
        elif input.dim == 3:
            m, n, k = input.shape

            grid_size = (
                cdiv(m, _BLOCK_SIZE_3D),
                cdiv(n, _BLOCK_SIZE_3D),
                cdiv(k, _BLOCK_SIZE_3D),
            )
            _uniary_with_scalar_3d_template[grid_size](
                input,
                scalar,
                output,
                m,
                n,
                k,
                *input.strides,
                *output.strides,
                BLOCK_M=_BLOCK_SIZE_3D,
                BLOCK_N=_BLOCK_SIZE_3D,
                BLOCK_K=_BLOCK_SIZE_3D,
                FN=fn,
            )
        elif input.dim == 4:
            d0, d1, d2, d3 = input.shape

            grid_size = (
                cdiv(d0, _BLOCK_SIZE_4D)
                * cdiv(d1, _BLOCK_SIZE_4D)
                * cdiv(d2, _BLOCK_SIZE_4D)
                * cdiv(d3, _BLOCK_SIZE_4D),
            )
            _uniary_with_scalar_4d_template[grid_size](
                input,
                scalar,
                output,
                d0,
                d1,
                d2,
                d3,
                *input.strides,
                *output.strides,
                BLOCK_D0=_BLOCK_SIZE_4D,
                BLOCK_D1=_BLOCK_SIZE_4D,
                BLOCK_D2=_BLOCK_SIZE_4D,
                BLOCK_D3=_BLOCK_SIZE_4D,
                FN=fn,
            )
        else:
            raise NotImplementedError(
                f"uniary_op not implemented for tensors with dim {input.dim}"
            )

        return output

    def uniary_op_inplace(input: _T, scalar: Scalar) -> _T:
        if input.dim == 1 or input.is_contiguous:
            numel = input.numel
            grid_size = (cdiv(numel, _BLOCK_SIZE_1D),)
            _uniary_with_scalar_1d_template[grid_size](
                in_ptr=input,
                scalar=scalar,
                out_ptr=input,
                numel=numel,
                BLOCK_SIZE=_BLOCK_SIZE_1D,
                FN=fn,
                INPLACE=True,
            )
        elif input.dim == 2:
            m, n = input.shape

            grid_size = (
                cdiv(m, _BLOCK_SIZE_2D),
                cdiv(n, _BLOCK_SIZE_2D),
            )
            _uniary_with_scalar_2d_template[grid_size](
                input,
                scalar,
                input,
                m,
                n,
                *input.strides,
                *input.strides,
                BLOCK_M=_BLOCK_SIZE_2D,
                BLOCK_N=_BLOCK_SIZE_2D,
                FN=fn,
                INPLACE=True,
            )
        elif input.dim == 3:
            m, n, k = input.shape

            grid_size = (
                cdiv(m, _BLOCK_SIZE_3D),
                cdiv(n, _BLOCK_SIZE_3D),
                cdiv(k, _BLOCK_SIZE_3D),
            )
            _uniary_with_scalar_3d_template[grid_size](
                input,
                scalar,
                input,
                m,
                n,
                k,
                *input.strides,
                *input.strides,
                BLOCK_M=_BLOCK_SIZE_3D,
                BLOCK_N=_BLOCK_SIZE_3D,
                BLOCK_K=_BLOCK_SIZE_3D,
                FN=fn,
                INPLACE=True,
            )
        elif input.dim == 4:
            d0, d1, d2, d3 = input.shape

            grid_size = (
                cdiv(d0, _BLOCK_SIZE_4D),
                cdiv(d1, _BLOCK_SIZE_4D),
                cdiv(d2, _BLOCK_SIZE_4D),
                cdiv(d3, _BLOCK_SIZE_4D),
            )
            _uniary_with_scalar_4d_template[grid_size](
                input,
                scalar,
                input,
                d0,
                d1,
                d2,
                d3,
                *input.strides,
                *input.strides,
                BLOCK_D0=_BLOCK_SIZE_4D,
                BLOCK_D1=_BLOCK_SIZE_4D,
                BLOCK_D2=_BLOCK_SIZE_4D,
                BLOCK_D3=_BLOCK_SIZE_4D,
                FN=fn,
                INPLACE=True,
            )
        else:
            raise NotImplementedError(
                f"uniary_op not implemented for tensors with dim {input.dim}"
            )

        return input

    return uniary_op_inplace if inplace else uniary_op_outplace


@jit
@no_type_check
def _replace(x, scalar):
    return scalar


@jit
@no_type_check
def _adds(x, scalar):
    return x + scalar


@jit
@no_type_check
def _subs(x, scalar):
    return x - scalar


@jit
@no_type_check
def _muls(x, scalar):
    return x * scalar


@jit
@no_type_check
def _divs(x, scalar):
    return x / scalar


@jit
@no_type_check
def _floordivs(x, scalar):
    return x // scalar


@jit
@no_type_check
def _mods(x, scalar):
    return x % scalar


adds = _generate_op(_adds, inplace=False)
adds_ = _generate_op(_adds, inplace=True)
subs = _generate_op(_subs, inplace=False)
subs_ = _generate_op(_subs, inplace=True)
muls = _generate_op(_muls, inplace=False)
muls_ = _generate_op(_muls, inplace=True)
divs = _generate_op(_divs, inplace=False)
divs_ = _generate_op(_divs, inplace=True)
floordivs = _generate_op(_floordivs, inplace=False)
floordivs_ = _generate_op(_floordivs, inplace=True)
mods = _generate_op(_mods, inplace=False)
mods_ = _generate_op(_mods, inplace=True)

fill_ = _generate_op(_replace, inplace=True)
