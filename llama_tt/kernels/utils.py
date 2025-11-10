from collections.abc import Sequence
from typing import Annotated, no_type_check

import triton.language as tl
from triton import jit


def broadcast_shapes(
    shape0: Sequence[int],
    shape1: Sequence[int],
    strides0: Sequence[int],
    strides1: Sequence[int],
) -> tuple[
    Annotated[Sequence[int], "broadcasted shape"],
    Annotated[Sequence[int], "broadcasted strides1"],
    Annotated[Sequence[int], "broadcasted strides2"],
]:
    """
    Raises:
        ValueError: if the two shapes are not broadcastable
    """

    if shape0 == shape1:
        return shape0, strides0, strides1

    ndim1 = len(shape0)
    ndim2 = len(shape1)
    ndim = max(ndim1, ndim2)

    shape = [-1] * ndim
    shape1_ = [1] * (ndim - ndim1) + list(shape0)
    shape2_ = [1] * (ndim - ndim2) + list(shape1)
    strides1_ = [0] * (ndim - ndim1) + list(strides0)
    strides2_ = [0] * (ndim - ndim2) + list(strides1)

    for i in range(ndim):
        dim1 = shape1_[i]
        dim2 = shape2_[i]
        if dim1 == dim2:
            shape[i] = dim1
        elif dim1 == 1:
            shape[i] = dim2
            strides1_[i] = 0
        elif dim2 == 1:
            shape[i] = dim1
            strides2_[i] = 0
        else:
            raise ValueError(f"Shapes {shape0} and {shape1} are not broadcastable")

    return shape, strides1_, strides2_


@jit
@no_type_check
def compute_pid4d(
    pid,
    num_pid_d0,
    num_pid_d1,
    num_pid_d2,
    num_pid_d3,
):
    pid0 = pid // (num_pid_d1 * num_pid_d2 * num_pid_d3)
    rem0 = pid % (num_pid_d1 * num_pid_d2 * num_pid_d3)
    pid1 = rem0 // (num_pid_d2 * num_pid_d3)
    rem1 = rem0 % (num_pid_d2 * num_pid_d3)
    pid2 = rem1 // num_pid_d3
    pid3 = rem1 % num_pid_d3
    return pid0, pid1, pid2, pid3


@jit
@no_type_check
def make_two_block_ptr(
    base1: tl.pointer_type,
    base2: tl.pointer_type,
    shape,
    strides1,
    strides2,
    offsets,
    block_shape: tl.constexpr,
    order: tl.constexpr,
):
    block_ptr1 = tl.make_block_ptr(
        base1,
        shape=shape,
        strides=strides1,
        offsets=offsets,
        block_shape=block_shape,
        order=order,
    )
    block_ptr2 = tl.make_block_ptr(
        base2,
        shape=shape,
        strides=strides2,
        offsets=offsets,
        block_shape=block_shape,
        order=order,
    )
    return block_ptr1, block_ptr2


@jit
@no_type_check
def make_three_block_ptr(
    base1: tl.pointer_type,
    base2: tl.pointer_type,
    base3: tl.pointer_type,
    shape,
    strides1,
    strides2,
    strides3,
    offsets,
    block_shape: tl.constexpr,
    order: tl.constexpr,
):
    block_ptr1 = tl.make_block_ptr(
        base1,
        shape=shape,
        strides=strides1,
        offsets=offsets,
        block_shape=block_shape,
        order=order,
    )
    block_ptr2 = tl.make_block_ptr(
        base2,
        shape=shape,
        strides=strides2,
        offsets=offsets,
        block_shape=block_shape,
        order=order,
    )
    block_ptr3 = tl.make_block_ptr(
        base3,
        shape=shape,
        strides=strides3,
        offsets=offsets,
        block_shape=block_shape,
        order=order,
    )
    return block_ptr1, block_ptr2, block_ptr3
