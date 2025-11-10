import logging
from typing import Literal

import colorlog

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
