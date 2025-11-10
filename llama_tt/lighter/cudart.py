from cuda.bindings.runtime import (cudaDeviceSynchronize, cudaError_t,
                                   cudaFreeAsync, cudaGetDevice,
                                   cudaGetDeviceCount, cudaGetDeviceProperties,
                                   cudaMallocAsync, cudaMemcpyAsync,
                                   cudaMemcpyKind, cudaSetDevice, cudaStream_t,
                                   cudaStreamCreate, cudaStreamDestroy,
                                   cudaStreamSynchronize)


def _check(err: cudaError_t) -> None:
    if err != cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error: {err}")


def _cuda_get_device() -> int:
    err, device = cudaGetDevice()
    _check(err)
    return device


def _cuda_set_device(device: int) -> None:
    (err,) = cudaSetDevice(device)
    _check(err)


def cuda_stream_create() -> cudaStream_t:
    err, stream = cudaStreamCreate()
    _check(err)
    return stream


def cuda_stream_destroy(stream: cudaStream_t) -> None:
    (err,) = cudaStreamDestroy(stream)
    _check(err)


def cuda_get_device_properties(device: int):
    err, props = cudaGetDeviceProperties(device)
    _check(err)
    return props


def cuda_get_device_count() -> int:
    err, count = cudaGetDeviceCount()
    _check(err)
    return count


# def _cuda_malloc(size: int) -> int:
#     err, ptr = cudaMalloc(size)
#     _check(err)
#     return ptr
#
#
# def _cuda_free(ptr: int) -> None:
#     err = cudaFree(ptr)
#     _check(err)


def _cuda_malloc_async(size: int, stream: cudaStream_t) -> int:
    err, ptr = cudaMallocAsync(size, stream)
    _check(err)
    return ptr


def _cuda_free_async(ptr: int, stream: cudaStream_t) -> None:
    (err,) = cudaFreeAsync(ptr, stream)
    _check(err)


def _cuda_memcpy_async(
    dst: int, src: int, size: int, kind: cudaMemcpyKind, stream: cudaStream_t
):
    (err,) = cudaMemcpyAsync(dst, src, size, kind, stream)
    _check(err)


def _cuda_device_synchronize():
    (err,) = cudaDeviceSynchronize()
    _check(err)


def _cuda_stream_synchronize(stream: cudaStream_t):
    (err,) = cudaStreamSynchronize(stream)
    _check(err)


# Global state


class Stream:
    def __init__(self, device: int | None = None):
        if device is None:
            device = current_device
        self._device: int = device
        self._stream: cudaStream_t = cuda_stream_create()
        self._previous_device: int
        self._previous_stream: Stream | None

    @property
    def stream(self) -> cudaStream_t:
        return self._stream

    def __int__(self) -> int:
        return int(self._stream)

    def __enter__(self):
        self._previous_device = current_device
        set_current_device(self._device)
        self._previous_stream = device2stream[self._device]
        device2stream[self._device] = self
        return self

    def __exit__(
        self,
        exc_type,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        exc_value,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        traceback,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    ):
        device2stream[self._device] = self._previous_stream
        set_current_device(self._previous_device)

    def __del__(self):
        cuda_stream_destroy(self._stream)


current_device = _cuda_get_device()
device2stream: dict[int, Stream | None] = {current_device: Stream()}


def set_current_device(device: int):
    global current_device
    if device != current_device:
        _cuda_set_device(device)
        current_device = device


def get_current_stream(device: int | None = None) -> Stream:
    if device is None:
        device = current_device
    stream = device2stream.get(device)
    if stream is None:
        stream = Stream(device)
        device2stream[device] = stream
    return stream


# Exposed memory management


def malloc(size: int) -> int:
    return _cuda_malloc_async(size, get_current_stream(current_device).stream)


def free(ptr: int) -> None:
    _cuda_free_async(ptr, get_current_stream(current_device).stream)


def memcpy(dst: int, src: int, size: int, kind: cudaMemcpyKind) -> None:
    _cuda_memcpy_async(
        dst,
        src,
        size,
        kind,
        get_current_stream(current_device).stream,
    )


def synchronize(stream: Stream | None = None) -> None:
    if stream is None:
        _cuda_device_synchronize()
    else:
        _cuda_stream_synchronize(stream.stream)
