from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Self

from llama_tt.lighter.dtype import dtype


class TensorBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def strides(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def numel(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> dtype: ...

    @property
    @abstractmethod
    def is_contiguous(self) -> bool: ...

    @abstractmethod
    def data_ptr(self) -> int: ...

    @abstractmethod
    def empty_like(self, contiguous: bool = False) -> Self: ...

    @classmethod
    @abstractmethod
    def empty(
        cls, shape: Sequence[int], dtype: "dtype", strides: Sequence[int] | None = None
    ) -> Self: ...

    @abstractmethod
    def new_empty(self, *size: int, dtype: "dtype | None" = None) -> Self: ...
