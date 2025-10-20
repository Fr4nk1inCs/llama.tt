import _cython_3_1_4
import functools
from typing import Any

DRIVER_CU_RESULT_EXPLANATIONS: dict
RUNTIME_CUDA_ERROR_EXPLANATIONS: dict
__pyx_capi__: dict
__test__: dict
cast_to_3_tuple: _cython_3_1_4.cython_function_or_method
check_or_create_options: _cython_3_1_4.cython_function_or_method
get_binding_version: functools._lru_cache_wrapper
handle_return: _cython_3_1_4.cython_function_or_method
is_nested_sequence: _cython_3_1_4.cython_function_or_method
is_sequence: _cython_3_1_4.cython_function_or_method
precondition: _cython_3_1_4.cython_function_or_method

class CUDAError(Exception): ...

class NVRTCError(CUDAError): ...

class Transaction:
    def __init__(self) -> Any: ...
    def append(self, fn, *args, **kwargs) -> Any: ...
    def commit(self) -> Any: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type, exc, tb) -> Any: ...
