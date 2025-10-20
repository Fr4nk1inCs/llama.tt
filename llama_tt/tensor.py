import logging
from collections.abc import Sequence
from functools import reduce
from typing import cast

import cuda.bindings.runtime as cudart

from llama_tt.dtype import Scalar, dtype
from llama_tt.utils import int_mul, with_check

logger = logging.getLogger(__name__)

CpuTensor = Sequence[Scalar] | Sequence["CpuTensor"]


class DeviceHandle:
    def __init__(self, ptr: int):
        self._ptr: int = ptr

    @property
    def ptr(self) -> int:
        return self._ptr

    @classmethod
    def alloc(cls, nbytes: int) -> "DeviceHandle":
        logger.debug("Allocating %d bytes of device memory", nbytes)
        ptr = cast(int, with_check(cudart.cudaMalloc)(nbytes))
        return cls(ptr)

    def __del__(self):
        logger.debug(f"Freeing device memory at ptr 0x%x", self._ptr)
        with_check(cudart.cudaFree)(self._ptr)


class DeviceTensor:
    def __init__(
        self,
        handle: DeviceHandle,
        shape: Sequence[int],
        dtype: dtype,
        strides: Sequence[int] | None = None,
    ):
        if strides is not None:
            if len(strides) != len(shape):
                raise ValueError(
                    f"Strides length {len(strides)} does not match shape length {len(shape)}"
                )
        else:
            strides = self._compute_default_strides(shape)

        self._handle: DeviceHandle = handle
        self._shape: Sequence[int] = shape
        self._dtype: "dtype" = dtype
        self._strides: Sequence[int] = strides
        self._numel: int = reduce(int_mul, shape, 1)

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

    def data_ptr(self) -> int:
        return self._handle.ptr

    @property
    def dtype(self) -> "dtype":
        return self._dtype

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def strides(self) -> Sequence[int]:
        return self._strides

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    def numel(self) -> int:
        return self._numel

    @classmethod
    def alloc(
        cls, shape: Sequence[int], dtype: "dtype", strides: Sequence[int] | None = None
    ) -> "DeviceTensor":
        numel: int = reduce(int_mul, shape, 1)
        nbytes: int = numel * dtype.bytes
        handle = DeviceHandle.alloc(nbytes)

        logger.debug(
            "Allocated DeviceTensor of shape %s, dtype %s, numel %d, nbytes %d",
            shape,
            dtype,
            numel,
            nbytes,
        )
        return cls(handle, shape, dtype, strides)

    def item(self):
        if self._numel != 1:
            raise ValueError("Can only call item() on tensors with one element")
        host_buf = bytes(self._dtype.bytes)
        with_check(cudart.cudaMemcpy)(
            host_buf,
            self.data_ptr(),
            self._dtype.bytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        return self._dtype.interpret_memory(host_buf)
