import _cython_3_1_4
from _typeshed import Incomplete
from cuda.core.experimental._memory import Buffer as Buffer
from typing import Any

__reduce_cython__: _cython_3_1_4.cython_function_or_method
__setstate_cython__: _cython_3_1_4.cython_function_or_method
__test__: dict

class ParamHolder:
    ptr: Incomplete
    def __init__(self, kernel_args) -> Any: ...
    def __reduce__(self): ...
