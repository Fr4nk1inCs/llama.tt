import logging
from typing import Any, Callable, Literal, ParamSpec, no_type_check

import colorlog
import cuda.bindings.runtime as cudart

_LogLevel = Literal[
    10,
    20,
    30,
    40,
    50,
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


def setup_logger(level: _LogLevel = logging.INFO, logfile: str | None = None):
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s %(levelname)8s] %(message)s %(fg_light_black)s(%(filename)s:%(lineno)d)%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "light_black",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bg_red,white",
        },
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    handlers: list[logging.Handler] = [console_handler]
    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_formatter = logging.Formatter(
            "[%(asctime)s %(levelname)8s] %(message)s (%(filename)s:%(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
    )


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
