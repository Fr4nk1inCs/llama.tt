from typing import no_type_check, override

from triton.backends import Backend, DriverBase, backends
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import CudaLauncher, CudaUtils, ty_to_cpp
from triton.testing import do_bench

from llama_tt.lighter import DeviceTensor, int32
from llama_tt.lighter.cudart import (cuda_get_device_count,
                                     cuda_get_device_properties,
                                     current_device, get_current_stream,
                                     set_current_device)


class CudaDriver(DriverBase):
    def __init__(self) -> None:
        self.utils: CudaUtils = CudaUtils()
        self.launcher_cls: type[CudaLauncher] = CudaLauncher
        super().__init__()

    def get_current_device(self) -> int:
        return current_device

    def set_current_device(self, device: int) -> None:
        set_current_device(device)

    def get_current_stream(self, device: int | None) -> int:
        return int(get_current_stream(device))

    def get_device_capability(self, device: int | None) -> tuple[int, int]:
        if device is None:
            device = current_device
        props = cuda_get_device_properties(device)
        return props.major, props.minor

    def get_empty_cache_for_benchmark(self):
        cache_size = 256 * 1024 * 1024  # 256MB
        return DeviceTensor.empty((cache_size // 4,), dtype=int32)

    def clear_cache(self, cache: DeviceTensor):
        _ = cache.fill_(0)

    @classmethod
    @no_type_check
    def is_active(self):
        device_count = cuda_get_device_count()
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
