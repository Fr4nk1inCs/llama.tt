import StridedMemoryView
import _cython_3_1_4
from _typeshed import Incomplete
from cuda.core.experimental._utils.cuda_utils import handle_return as handle_return
from typing import Any, ClassVar

__reduce_cython__: _cython_3_1_4.cython_function_or_method
__setstate_cython__: _cython_3_1_4.cython_function_or_method
__test__: dict
args_viewable_as_strided_memory: _cython_3_1_4.cython_function_or_method
view_as_cai: _cython_3_1_4.cython_function_or_method

class StridedMemoryView:
    device_id: Incomplete
    dtype: StridedMemoryView.dtype
    exporting_obj: Incomplete
    is_device_accessible: Incomplete
    ptr: Incomplete
    readonly: Incomplete
    shape: StridedMemoryView.shape
    strides: StridedMemoryView.strides
    def __init__(self, obj=..., stream_ptr=...) -> Any: ...
    def __reduce__(self): ...

class _StridedMemoryViewProxy:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, obj) -> Any: ...
    def view(self, stream_ptr=...) -> StridedMemoryView: ...
    def __reduce__(self): ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...
