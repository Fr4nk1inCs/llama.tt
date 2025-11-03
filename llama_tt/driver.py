from typing import no_type_check, override

import cuda.bindings.runtime as cudart
from triton.backends import Backend, DriverBase, backends
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import CudaLauncher, CudaUtils, ty_to_cpp
from triton.testing import do_bench

from llama_tt.lighter import DeviceTensor, int32
from llama_tt.utils import with_check


class MockTorchDevice:
    def __init__(self, type: str, index: int) -> None:
        self.type: str = type
        self.index: int = index


class CudaDriver(DriverBase):
    def __init__(self) -> None:
        self.utils: CudaUtils = CudaUtils()
        self.launcher_cls: type[CudaLauncher] = CudaLauncher

        self.device_to_stream: dict[int, int] = {}

        super().__init__()

    def get_current_device(self) -> int:
        return with_check(cudart.cudaGetDevice)()

    def set_current_device(self, device: int) -> None:
        with_check(cudart.cudaSetDevice)(device)

    def get_current_stream(self, device: int | None) -> int:
        if device is None:
            device = self.get_current_device()
        if device not in self.device_to_stream:
            stream = with_check(cudart.cudaStreamCreateWithFlags)(
                cudart.cudaStreamDefault
            )
            self.device_to_stream[device] = int(stream)
        return self.device_to_stream[device]

    def get_device_capability(self, device: int | None) -> tuple[int, int]:
        if device is None:
            device = self.get_current_device()
        props = with_check(cudart.cudaGetDeviceProperties)(device)
        return props.major, props.minor

    def get_empty_cache_for_benchmark(self):
        cache_size = 256 * 1024 * 1024  # 256MB
        return DeviceTensor.empty((cache_size // 4,), dtype=int32)

    def clear_cache(self, cache: DeviceTensor):
        cache.fill_(0)

    @classmethod
    @no_type_check
    def is_active(self):
        device_count = with_check(cudart.cudaGetDeviceCount)()
        return device_count > 0

    @override
    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    @override
    @no_type_check
    def get_current_target(self):
        device = self.get_current_device()
        major, minor = self.get_device_capability(device)
        capability = major * 10 + minor
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    @override
    @no_type_check
    def get_benchmarker(self):
        return do_bench

    @override
    def get_active_torch_device(self):
        pass


def setup_driver():
    old_nvidia_backend = backends["nvidia"]
    backends["nvidia"] = Backend(
        compiler=old_nvidia_backend.compiler,
        driver=CudaDriver,
    )
