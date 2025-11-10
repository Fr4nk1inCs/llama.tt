import logging

import numpy as np

from llama_tt import lighter
from llama_tt.driver import setup_driver
from llama_tt.log import setup_logger

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logger(logging.DEBUG)
    setup_driver()

    a_cpu = np.random.rand(1024, 1024).astype(np.float32)
    b_cpu = np.random.rand(1024, 1024).astype(np.float32)

    a_dev = lighter.from_numpy(a_cpu)
    b_dev = lighter.from_numpy(b_cpu)

    c_cpu = a_cpu + b_cpu
    c_dev = a_dev + b_dev

    c_dev_host = c_dev.numpy()

    np.testing.assert_allclose(c_dev_host, c_cpu)

    logger.info("Test passed!")
