from collections.abc import Sequence

from llama_tt.lighter.dtype import CorrespondingNDArrays, dtype
from llama_tt.lighter.tensor import DeviceTensor, Scalar


def empty(
    shape: Sequence[int], dtype: dtype, strides: Sequence[int] | None = None
) -> "DeviceTensor":
    return DeviceTensor.empty(shape, dtype, strides)


def empty_like(tensor: DeviceTensor) -> DeviceTensor:
    return DeviceTensor.empty(tensor.shape, tensor.dtype, tensor.strides)


def from_numpy(array: CorrespondingNDArrays) -> DeviceTensor:
    return DeviceTensor.from_numpy(array)


def fill(tensor: DeviceTensor, value: Scalar) -> DeviceTensor:
    return DeviceTensor.fill_(tensor, value)


def add(a: DeviceTensor, b: DeviceTensor) -> DeviceTensor:
    return a + b
