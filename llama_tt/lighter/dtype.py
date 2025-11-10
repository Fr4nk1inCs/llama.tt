from typing import override

import numpy as np
import numpy.typing as npt
import triton.language as tl

npdtype_mapping: dict[npt.DTypeLike, "dtype"] = {}
tldtype_mapping: dict[tl.dtype, "dtype"] = {}


class dtype:
    def __init__(
        self,
        name: str,
        nbytes: int,
        triton_dtype: tl.dtype,
        numpy_dtype: npt.DTypeLike | None,
    ):
        self._name: str = name
        self._bytes: int = nbytes
        self._triton_dtype: tl.dtype = triton_dtype
        self._numpy_dtype: npt.DTypeLike | None = numpy_dtype

        global npdtype_mapping, tldtype_mapping
        if numpy_dtype is not None:
            npdtype_mapping[np.dtype(numpy_dtype)] = self
            npdtype_mapping[numpy_dtype] = self
        tldtype_mapping[triton_dtype] = self

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
    def numpy_dtype(self) -> npt.DTypeLike | None:
        return self._numpy_dtype

    @override
    def __str__(self) -> str:
        return self._name


bfloat16 = dtype("bfloat16", 2, tl.bfloat16, None)
float16 = dtype("float16", 2, tl.float16, np.float16)
float32 = dtype("float32", 4, tl.float32, np.float32)
int32 = dtype("int32", 4, tl.int32, np.int32)
bool_ = dtype("bool", 1, tl.int1, np.bool_)

CorrespondingNDArrays = (
    npt.NDArray[np.float32] | npt.NDArray[np.int32] | npt.NDArray[np.bool_]
)
Scalar = int | float | bool
