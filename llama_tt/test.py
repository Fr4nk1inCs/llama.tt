import logging

from llama_tt.driver import setup_driver
from llama_tt.dtype import float32
from llama_tt.ops.elemwise import add, fill
from llama_tt.ops.reduce import all_eq
from llama_tt.tensor import DeviceTensor
from llama_tt.utils import setup_logger

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logger(logging.DEBUG)
    setup_driver()

    a = DeviceTensor.alloc([1024, 512], float32)
    b = DeviceTensor.alloc([1024, 512], float32)

    fill(a, 1.0)
    fill(b, 2.0)

    c = add(a, b)

    if all_eq(c, 3.0):
        logger.info("Test passed: all elements in c are equal to 3.0")
    else:
        logger.error("Test failed: not all elements in c are equal to 3.0")
