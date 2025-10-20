import struct
from typing import Callable, TypeVar, override

import numpy as np
import numpy.typing as npt
import triton.language as tl

Scalar = int | float | bool


class dtype:
    def __init__(
        self,
        name: str,
        nbytes: int,
        triton_dtype: tl.dtype,
        numpy_dtype: npt.DTypeLike,
        interpret_memory: Callable[[bytes], Scalar],
    ):
        self._name: str = name
        self._bytes: int = nbytes
        self._triton_dtype: tl.dtype = triton_dtype
        self._numpy_dtype: npt.DTypeLike = numpy_dtype
        self.interpret_memory: Callable[[bytes], Scalar] = interpret_memory

    @property
    def name(self) -> str:
        return self._name

    @property
    def bytes(self) -> int:
        return self._bytes

    @property
    def triton_dtype(self) -> tl.dtype:
        return self._triton_dtype

    @property
    def numpy_dtype(self) -> npt.DTypeLike:
        return self._numpy_dtype

    @override
    def __str__(self) -> str:
        return self._name


float32 = dtype(
    "float32", 4, tl.float32, np.float32, lambda b: struct.unpack("f", b)[0]
)
int32 = dtype("int32", 4, tl.int32, np.int32, lambda b: struct.unpack("i", b)[0])
bool_ = dtype("bool", 1, tl.int1, np.bool_, lambda b: bool(int.from_bytes(b, "little")))
