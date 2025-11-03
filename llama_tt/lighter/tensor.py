import logging
from collections.abc import Sequence
from functools import cache, reduce
from typing import Self, cast

import cuda.bindings.runtime as cudart
import numpy as np

from llama_tt.kernels.elemwise import add_kernel, fill_kernel
from llama_tt.lighter.dtype import (CorrespondingNDArrays, dtype,
                                    npdtype_mapping)
from llama_tt.utils import cdiv, int_mul, with_check

logger = logging.getLogger(__name__)

Scalar = int | float | bool


class DeviceHandle:
    def __init__(self, ptr: int):
        self._ptr: int = ptr

    @property
    def ptr(self) -> int:
        return self._ptr

    @classmethod
    def allocate(cls, nbytes: int) -> "DeviceHandle":
        logger.debug("Allocating %d bytes of device memory", nbytes)
        ptr = cast(int, with_check(cudart.cudaMalloc)(nbytes))
        return cls(ptr)

    def __del__(self):
        logger.debug("Freeing device memory at ptr 0x%x", self._ptr)
        with_check(cudart.cudaFree)(self._ptr)


class DevicePointer:
    def __init__(
        self,
        handle: DeviceHandle,
        offset: int = 0,
    ):
        self._handle: DeviceHandle = handle
        self._offset: int = offset

    @property
    @cache
    def ptr(self) -> int:
        return self._handle.ptr + self._offset

    def offset(self, offset: int) -> "DevicePointer":
        return DevicePointer(self._handle, self._offset + offset)

    @classmethod
    def allocate(cls, nbytes: int) -> "DevicePointer":
        handle = DeviceHandle.allocate(nbytes)
        return cls(handle)


class DeviceTensor:
    """
    A tensor is a multi-dimensional array stored on a contiguous block of device
    memory. While the underlying memory is contiguous, the tensor can have
    arbitrary shape and strides, making the actual data layout non-contiguous.
    """

    def __init__(
        self,
        handle: DeviceHandle,
        shape: Sequence[int],
        dtype: dtype,
        strides: Sequence[int] | None = None,
    ):
        self._handle: DeviceHandle = handle
        self._shape: Sequence[int] = shape
        self._dtype: "dtype" = dtype
        self._numel: int = reduce(int_mul, shape, 1)
        self._strides: Sequence[int]
        self._is_contiguous: bool

        if strides is not None:
            if len(strides) != len(shape):
                raise ValueError(
                    f"Strides length {len(strides)} does not match shape length {len(shape)}"
                )
            self._strides = strides
            self._is_contiguous = self._check_contiguous()
        else:
            self._strides = self._compute_default_strides(shape)
            self._is_contiguous = True

        self._numpied: CorrespondingNDArrays | None = None

    def _compute_default_strides(
        self,
        shape: Sequence[int],
    ) -> Sequence[int]:
        strides: list[int] = []
        stride: int = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return strides

    def _check_contiguous(self) -> bool:
        expected_stride = 1
        for dim in reversed(range(len(self.shape))):
            if self.stride(dim) != expected_stride:
                return False
            expected_stride *= self.shape[dim]
        return True

    def data_ptr(self) -> int:
        return self._handle.ptr

    @property
    def dtype(self) -> "dtype":
        return self._dtype

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def numel(self) -> int:
        return self._numel

    @property
    def strides(self) -> Sequence[int]:
        return self._strides

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    @property
    def is_contiguous(self) -> bool:
        return self._is_contiguous

    """Creation"""

    @classmethod
    def empty(
        cls, shape: Sequence[int], dtype: "dtype", strides: Sequence[int] | None = None
    ) -> "DeviceTensor":
        logger.debug(
            "Creating DeviceTensor of shape %s, dtype %s, strides %s",
            shape,
            dtype,
            strides,
        )

        numel: int = reduce(int_mul, shape, 1)
        nbytes: int = numel * dtype.bytes
        handle: DeviceHandle = DeviceHandle.allocate(nbytes)
        return cls(handle, shape, dtype, strides)

    """Data Transfer"""

    @classmethod
    def from_numpy(cls, array: CorrespondingNDArrays) -> "DeviceTensor":
        shape = array.shape
        dtype = npdtype_mapping[array.dtype]
        itemsize = array.dtype.itemsize
        strides = tuple(s // itemsize for s in array.strides)
        tensor = cls.empty(shape, dtype, strides)
        with_check(cudart.cudaMemcpy)(
            tensor.data_ptr(),
            array.ctypes.data,
            array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        return tensor

    def numpy(self) -> CorrespondingNDArrays:
        if self._numpied is not None:
            return self._numpied

        assert self.is_contiguous, "numpy() only supports contiguous tensors"
        nbytes = self.numel * self.dtype.bytes
        self._numpied = np.empty(self.shape, dtype=self.dtype.numpy_dtype)
        with_check(cudart.cudaMemcpy)(
            self._numpied.ctypes.data,
            self.data_ptr(),
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        return self._numpied

    def item(self):
        if self._numel != 1:
            raise ValueError("Can only call item() on tensors with one element")
        return self.numpy().item()

    def __repr__(self) -> str:
        return (
            f"DeviceTensor({np.array2string(self.numpy())}, dtype={self.dtype}, "
            f"shape={self.shape}, strides={self.strides})"
        )

    """Conversions"""

    """In-place operations"""

    def fill_(self, value: Scalar) -> Self:
        assert self.is_contiguous, "fill_ only supports contiguous tensors"
        self._numpied = None  # Invalidate cached numpy array

        BLOCK_SIZE = 1024
        numel = self.numel
        grid = (cdiv(numel, BLOCK_SIZE),)
        fill_kernel[grid](self, value, numel, BLOCK_SIZE=1024)
        return self

    """Computation"""

    def __add__(self, other: "DeviceTensor") -> "DeviceTensor":
        assert self.shape == other.shape, "Shapes must match for addition"
        assert self.strides == other.strides, "Strides must match for addition"

        output = DeviceTensor.empty(self.shape, self.dtype, self.strides)
        BLOCK_SIZE = 1024
        numel = self.numel
        grid = (cdiv(numel, BLOCK_SIZE),)
        add_kernel[grid](self, other, output, numel, BLOCK_SIZE=BLOCK_SIZE)
        return output
