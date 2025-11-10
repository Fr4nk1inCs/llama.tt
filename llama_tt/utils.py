from typing import Any, Callable, Literal, ParamSpec, no_type_check

import cuda.bindings.runtime as cudart

def int_mul(a: int, b: int) -> int:
    return a * b


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


Spec = ParamSpec("Spec")


def with_check(func: Callable[Spec, Any], /) -> Callable[Spec, Any]:
    @no_type_check
    def wrapped(*args: Spec.args, **kwargs: Spec.kwargs) -> Any:
        err, *rest = func(*args, **kwargs)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Error: {err}")
        if len(rest) == 0:
            return
        if len(rest) == 1:
            return rest[0]
        return tuple(rest)

    return wrapped


Todo = NotImplementedError
