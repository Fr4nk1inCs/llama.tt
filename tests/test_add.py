import numpy as np
import pytest

from llama_tt import lighter
from llama_tt.lighter.driver import setup_driver
from llama_tt.lighter.dtype import npdtype_mapping


@pytest.mark.parametrize("dtype", [lighter.float32, lighter.float16, lighter.bfloat16])
def test_add1d(dtype: lighter.dtype):
    setup_driver()

    np_dtype = dtype.numpy_dtype if dtype != lighter.bfloat16 else np.float32

    in0_cpu = np.random.rand(16384).astype(np_dtype)
    in1_cpu = np.random.rand(16384).astype(np_dtype)
    out_cpu = in0_cpu + in1_cpu

    in0_dev = lighter.from_numpy(in0_cpu).to_dtype(dtype)
    in1_dev = lighter.from_numpy(in1_cpu).to_dtype(dtype)
    out_dev = lighter.add(in0_dev, in1_dev).to_dtype(npdtype_mapping[np_dtype])

    out_dev_cpu = out_dev.numpy()

    if dtype == lighter.bfloat16:
        np.testing.assert_allclose(out_dev_cpu, out_cpu, rtol=1e-2, atol=1e-2)
    else:
        np.testing.assert_allclose(out_dev_cpu, out_cpu)


@pytest.mark.parametrize("dtype", [lighter.float32, lighter.float16, lighter.bfloat16])
def test_add2d(dtype: lighter.dtype):
    setup_driver()

    np_dtype = dtype.numpy_dtype if dtype != lighter.bfloat16 else np.float32

    in0_cpu = np.random.rand(256, 256).astype(np_dtype)
    in1_cpu = np.random.rand(256, 256).astype(np_dtype)
    out_cpu = in0_cpu + in1_cpu

    in0_dev = lighter.from_numpy(in0_cpu).to_dtype(dtype)
    in1_dev = lighter.from_numpy(in1_cpu).to_dtype(dtype)
    out_dev = lighter.add(in0_dev, in1_dev).to_dtype(npdtype_mapping[np_dtype])

    out_dev_cpu = out_dev.numpy()

    if dtype == lighter.bfloat16:
        np.testing.assert_allclose(out_dev_cpu, out_cpu, rtol=1e-2, atol=1e-2)
    else:
        np.testing.assert_allclose(out_dev_cpu, out_cpu)


@pytest.mark.parametrize("dtype", [lighter.float32, lighter.float16, lighter.bfloat16])
def test_add3d(dtype: lighter.dtype):
    setup_driver()

    np_dtype = dtype.numpy_dtype if dtype != lighter.bfloat16 else np.float32

    in0_cpu = np.random.rand(32, 32, 32).astype(np_dtype)
    in1_cpu = np.random.rand(32, 32, 32).astype(np_dtype)
    out_cpu = in0_cpu + in1_cpu

    in0_dev = lighter.from_numpy(in0_cpu).to_dtype(dtype)
    in1_dev = lighter.from_numpy(in1_cpu).to_dtype(dtype)
    out_dev = lighter.add(in0_dev, in1_dev).to_dtype(npdtype_mapping[np_dtype])

    out_dev_cpu = out_dev.numpy()

    if dtype == lighter.bfloat16:
        np.testing.assert_allclose(out_dev_cpu, out_cpu, rtol=1e-2, atol=1e-2)
    else:
        np.testing.assert_allclose(out_dev_cpu, out_cpu)


@pytest.mark.parametrize("dtype", [lighter.float32, lighter.float16, lighter.bfloat16])
def test_add4d(dtype: lighter.dtype):
    setup_driver()

    np_dtype = dtype.numpy_dtype if dtype != lighter.bfloat16 else np.float32

    in0_cpu = np.random.rand(16, 16, 16, 16).astype(np_dtype)
    in1_cpu = np.random.rand(16, 16, 16, 16).astype(np_dtype)
    out_cpu = in0_cpu + in1_cpu

    in0_dev = lighter.from_numpy(in0_cpu).to_dtype(dtype)
    in1_dev = lighter.from_numpy(in1_cpu).to_dtype(dtype)
    out_dev = lighter.add(in0_dev, in1_dev).to_dtype(npdtype_mapping[np_dtype])

    out_dev_cpu = out_dev.numpy()

    if dtype == lighter.bfloat16:
        np.testing.assert_allclose(out_dev_cpu, out_cpu, rtol=1e-2, atol=1e-2)
    else:
        np.testing.assert_allclose(out_dev_cpu, out_cpu)
