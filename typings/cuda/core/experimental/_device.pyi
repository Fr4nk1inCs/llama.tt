import DeviceProperties
from _frozen_importlib import ComputeCapability
from cuda.core.experimental._context import Context as Context, ContextOptions as ContextOptions
from cuda.core.experimental._event import Event as Event, EventOptions as EventOptions
from cuda.core.experimental._graph import GraphBuilder as GraphBuilder
from cuda.core.experimental._memory import Buffer as Buffer, DeviceMemoryResource as DeviceMemoryResource, MemoryResource as MemoryResource
from cuda.core.experimental._stream import IsStreamT as IsStreamT, Stream as Stream, StreamOptions as StreamOptions
from cuda.core.experimental._utils.clear_error_support import assert_type as assert_type
from cuda.core.experimental._utils.cuda_utils import CUDAError as CUDAError, handle_return as handle_return
from typing import Any, ClassVar

__test__: dict

class Device:
    memory_resource: MemoryResource
    def __init__(self, *args, **kwargs) -> None: ...
    def allocate(self, size, stream: Stream | None = ...) -> Buffer: ...
    def create_context(self, options: ContextOptions = ...) -> Context: ...
    def create_event(self, options: EventOptions | None = ...) -> Event: ...
    def create_graph_builder(self) -> GraphBuilder: ...
    def create_stream(self, obj: IsStreamT | None = ..., options: StreamOptions | None = ...) -> Stream: ...
    def set_current(self, ctx: Context = ...) -> Context | None: ...
    def sync(self) -> Any: ...
    def __int__(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    @property
    def arch(self) -> str: ...
    @property
    def compute_capability(self) -> ComputeCapability: ...
    @property
    def context(self) -> Context: ...
    @property
    def default_stream(self) -> Stream: ...
    @property
    def device_id(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def pci_bus_id(self) -> str: ...
    @property
    def properties(self) -> DeviceProperties: ...
    @property
    def uuid(self) -> str: ...

class DeviceProperties:
    _init: ClassVar[method] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    async_engine_count: DeviceProperties.async_engine_count
    can_flush_remote_writes: DeviceProperties.can_flush_remote_writes
    can_map_host_memory: DeviceProperties.can_map_host_memory
    can_tex2d_gather: DeviceProperties.can_tex2d_gather
    can_use_64_bit_stream_mem_ops: DeviceProperties.can_use_64_bit_stream_mem_ops
    can_use_host_pointer_for_registered_mem: DeviceProperties.can_use_host_pointer_for_registered_mem
    can_use_stream_wait_value_nor: DeviceProperties.can_use_stream_wait_value_nor
    clock_rate: DeviceProperties.clock_rate
    cluster_launch: DeviceProperties.cluster_launch
    compute_capability_major: DeviceProperties.compute_capability_major
    compute_capability_minor: DeviceProperties.compute_capability_minor
    compute_mode: DeviceProperties.compute_mode
    compute_preemption_supported: DeviceProperties.compute_preemption_supported
    concurrent_kernels: DeviceProperties.concurrent_kernels
    concurrent_managed_access: DeviceProperties.concurrent_managed_access
    cooperative_launch: DeviceProperties.cooperative_launch
    d3d12_cig_supported: DeviceProperties.d3d12_cig_supported
    deferred_mapping_cuda_array_supported: DeviceProperties.deferred_mapping_cuda_array_supported
    direct_managed_mem_access_from_host: DeviceProperties.direct_managed_mem_access_from_host
    dma_buf_supported: DeviceProperties.dma_buf_supported
    ecc_enabled: DeviceProperties.ecc_enabled
    generic_compression_supported: DeviceProperties.generic_compression_supported
    global_l1_cache_supported: DeviceProperties.global_l1_cache_supported
    global_memory_bus_width: DeviceProperties.global_memory_bus_width
    gpu_direct_rdma_flush_writes_options: DeviceProperties.gpu_direct_rdma_flush_writes_options
    gpu_direct_rdma_supported: DeviceProperties.gpu_direct_rdma_supported
    gpu_direct_rdma_with_cuda_vmm_supported: DeviceProperties.gpu_direct_rdma_with_cuda_vmm_supported
    gpu_direct_rdma_writes_ordering: DeviceProperties.gpu_direct_rdma_writes_ordering
    gpu_overlap: DeviceProperties.gpu_overlap
    gpu_pci_device_id: DeviceProperties.gpu_pci_device_id
    gpu_pci_subsystem_id: DeviceProperties.gpu_pci_subsystem_id
    handle_type_fabric_supported: DeviceProperties.handle_type_fabric_supported
    handle_type_posix_file_descriptor_supported: DeviceProperties.handle_type_posix_file_descriptor_supported
    handle_type_win32_handle_supported: DeviceProperties.handle_type_win32_handle_supported
    handle_type_win32_kmt_handle_supported: DeviceProperties.handle_type_win32_kmt_handle_supported
    host_alloc_dma_buf_supported: DeviceProperties.host_alloc_dma_buf_supported
    host_memory_pools_supported: DeviceProperties.host_memory_pools_supported
    host_native_atomic_supported: DeviceProperties.host_native_atomic_supported
    host_numa_id: DeviceProperties.host_numa_id
    host_numa_memory_pools_supported: DeviceProperties.host_numa_memory_pools_supported
    host_numa_multinode_ipc_supported: DeviceProperties.host_numa_multinode_ipc_supported
    host_numa_virtual_memory_management_supported: DeviceProperties.host_numa_virtual_memory_management_supported
    host_register_supported: DeviceProperties.host_register_supported
    host_virtual_memory_management_supported: DeviceProperties.host_virtual_memory_management_supported
    integrated: DeviceProperties.integrated
    ipc_event_supported: DeviceProperties.ipc_event_supported
    kernel_exec_timeout: DeviceProperties.kernel_exec_timeout
    l2_cache_size: DeviceProperties.l2_cache_size
    local_l1_cache_supported: DeviceProperties.local_l1_cache_supported
    managed_memory: DeviceProperties.managed_memory
    max_access_policy_window_size: DeviceProperties.max_access_policy_window_size
    max_block_dim_x: DeviceProperties.max_block_dim_x
    max_block_dim_y: DeviceProperties.max_block_dim_y
    max_block_dim_z: DeviceProperties.max_block_dim_z
    max_blocks_per_multiprocessor: DeviceProperties.max_blocks_per_multiprocessor
    max_grid_dim_x: DeviceProperties.max_grid_dim_x
    max_grid_dim_y: DeviceProperties.max_grid_dim_y
    max_grid_dim_z: DeviceProperties.max_grid_dim_z
    max_persisting_l2_cache_size: DeviceProperties.max_persisting_l2_cache_size
    max_pitch: DeviceProperties.max_pitch
    max_registers_per_block: DeviceProperties.max_registers_per_block
    max_registers_per_multiprocessor: DeviceProperties.max_registers_per_multiprocessor
    max_shared_memory_per_block: DeviceProperties.max_shared_memory_per_block
    max_shared_memory_per_block_optin: DeviceProperties.max_shared_memory_per_block_optin
    max_shared_memory_per_multiprocessor: DeviceProperties.max_shared_memory_per_multiprocessor
    max_threads_per_block: DeviceProperties.max_threads_per_block
    max_threads_per_multiprocessor: DeviceProperties.max_threads_per_multiprocessor
    maximum_surface1d_layered_layers: DeviceProperties.maximum_surface1d_layered_layers
    maximum_surface1d_layered_width: DeviceProperties.maximum_surface1d_layered_width
    maximum_surface1d_width: DeviceProperties.maximum_surface1d_width
    maximum_surface2d_height: DeviceProperties.maximum_surface2d_height
    maximum_surface2d_layered_height: DeviceProperties.maximum_surface2d_layered_height
    maximum_surface2d_layered_layers: DeviceProperties.maximum_surface2d_layered_layers
    maximum_surface2d_layered_width: DeviceProperties.maximum_surface2d_layered_width
    maximum_surface2d_width: DeviceProperties.maximum_surface2d_width
    maximum_surface3d_depth: DeviceProperties.maximum_surface3d_depth
    maximum_surface3d_height: DeviceProperties.maximum_surface3d_height
    maximum_surface3d_width: DeviceProperties.maximum_surface3d_width
    maximum_surfacecubemap_layered_layers: DeviceProperties.maximum_surfacecubemap_layered_layers
    maximum_surfacecubemap_layered_width: DeviceProperties.maximum_surfacecubemap_layered_width
    maximum_surfacecubemap_width: DeviceProperties.maximum_surfacecubemap_width
    maximum_texture1d_layered_layers: DeviceProperties.maximum_texture1d_layered_layers
    maximum_texture1d_layered_width: DeviceProperties.maximum_texture1d_layered_width
    maximum_texture1d_linear_width: DeviceProperties.maximum_texture1d_linear_width
    maximum_texture1d_mipmapped_width: DeviceProperties.maximum_texture1d_mipmapped_width
    maximum_texture1d_width: DeviceProperties.maximum_texture1d_width
    maximum_texture2d_gather_height: DeviceProperties.maximum_texture2d_gather_height
    maximum_texture2d_gather_width: DeviceProperties.maximum_texture2d_gather_width
    maximum_texture2d_height: DeviceProperties.maximum_texture2d_height
    maximum_texture2d_layered_height: DeviceProperties.maximum_texture2d_layered_height
    maximum_texture2d_layered_layers: DeviceProperties.maximum_texture2d_layered_layers
    maximum_texture2d_layered_width: DeviceProperties.maximum_texture2d_layered_width
    maximum_texture2d_linear_height: DeviceProperties.maximum_texture2d_linear_height
    maximum_texture2d_linear_pitch: DeviceProperties.maximum_texture2d_linear_pitch
    maximum_texture2d_linear_width: DeviceProperties.maximum_texture2d_linear_width
    maximum_texture2d_mipmapped_height: DeviceProperties.maximum_texture2d_mipmapped_height
    maximum_texture2d_mipmapped_width: DeviceProperties.maximum_texture2d_mipmapped_width
    maximum_texture2d_width: DeviceProperties.maximum_texture2d_width
    maximum_texture3d_depth: DeviceProperties.maximum_texture3d_depth
    maximum_texture3d_depth_alternate: DeviceProperties.maximum_texture3d_depth_alternate
    maximum_texture3d_height: DeviceProperties.maximum_texture3d_height
    maximum_texture3d_height_alternate: DeviceProperties.maximum_texture3d_height_alternate
    maximum_texture3d_width: DeviceProperties.maximum_texture3d_width
    maximum_texture3d_width_alternate: DeviceProperties.maximum_texture3d_width_alternate
    maximum_texturecubemap_layered_layers: DeviceProperties.maximum_texturecubemap_layered_layers
    maximum_texturecubemap_layered_width: DeviceProperties.maximum_texturecubemap_layered_width
    maximum_texturecubemap_width: DeviceProperties.maximum_texturecubemap_width
    mem_decompress_algorithm_mask: DeviceProperties.mem_decompress_algorithm_mask
    mem_decompress_maximum_length: DeviceProperties.mem_decompress_maximum_length
    mem_sync_domain_count: DeviceProperties.mem_sync_domain_count
    memory_clock_rate: DeviceProperties.memory_clock_rate
    memory_pools_supported: DeviceProperties.memory_pools_supported
    mempool_supported_handle_types: DeviceProperties.mempool_supported_handle_types
    mps_enabled: DeviceProperties.mps_enabled
    multi_gpu_board: DeviceProperties.multi_gpu_board
    multi_gpu_board_group_id: DeviceProperties.multi_gpu_board_group_id
    multicast_supported: DeviceProperties.multicast_supported
    multiprocessor_count: DeviceProperties.multiprocessor_count
    numa_config: DeviceProperties.numa_config
    numa_id: DeviceProperties.numa_id
    only_partial_host_native_atomic_supported: DeviceProperties.only_partial_host_native_atomic_supported
    pageable_memory_access: DeviceProperties.pageable_memory_access
    pageable_memory_access_uses_host_page_tables: DeviceProperties.pageable_memory_access_uses_host_page_tables
    pci_bus_id: DeviceProperties.pci_bus_id
    pci_device_id: DeviceProperties.pci_device_id
    pci_domain_id: DeviceProperties.pci_domain_id
    read_only_host_register_supported: DeviceProperties.read_only_host_register_supported
    reserved_shared_memory_per_block: DeviceProperties.reserved_shared_memory_per_block
    single_to_double_precision_perf_ratio: DeviceProperties.single_to_double_precision_perf_ratio
    sparse_cuda_array_supported: DeviceProperties.sparse_cuda_array_supported
    stream_priorities_supported: DeviceProperties.stream_priorities_supported
    surface_alignment: DeviceProperties.surface_alignment
    tcc_driver: DeviceProperties.tcc_driver
    tensor_map_access_supported: DeviceProperties.tensor_map_access_supported
    texture_alignment: DeviceProperties.texture_alignment
    texture_pitch_alignment: DeviceProperties.texture_pitch_alignment
    timeline_semaphore_interop_supported: DeviceProperties.timeline_semaphore_interop_supported
    total_constant_memory: DeviceProperties.total_constant_memory
    unified_addressing: DeviceProperties.unified_addressing
    unified_function_pointers: DeviceProperties.unified_function_pointers
    virtual_memory_management_supported: DeviceProperties.virtual_memory_management_supported
    vulkan_cig_supported: DeviceProperties.vulkan_cig_supported
    warp_size: DeviceProperties.warp_size
    def __init__(self, *args, **kwargs) -> Any: ...
    def __reduce__(self): ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...
