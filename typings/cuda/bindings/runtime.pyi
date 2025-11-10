from enum import IntEnum, auto

class cudaError_t(IntEnum):
    cudaErrorAddressOfConstant = auto()
    cudaErrorAlreadyAcquired = auto()
    cudaErrorAlreadyMapped = auto()
    cudaErrorApiFailureBase = auto()
    cudaErrorArrayIsMapped = auto()
    cudaErrorAssert = auto()
    cudaErrorCallRequiresNewerDriver = auto()
    cudaErrorCapturedEvent = auto()
    cudaErrorCdpNotSupported = auto()
    cudaErrorCdpVersionMismatch = auto()
    cudaErrorCompatNotSupportedOnDevice = auto()
    cudaErrorContained = auto()
    cudaErrorContextIsDestroyed = auto()
    cudaErrorCooperativeLaunchTooLarge = auto()
    cudaErrorCudartUnloading = auto()
    cudaErrorDeviceAlreadyInUse = auto()
    cudaErrorDeviceNotLicensed = auto()
    cudaErrorDeviceUninitialized = auto()
    cudaErrorDevicesUnavailable = auto()
    cudaErrorDuplicateSurfaceName = auto()
    cudaErrorDuplicateTextureName = auto()
    cudaErrorDuplicateVariableName = auto()
    cudaErrorECCUncorrectable = auto()
    cudaErrorExternalDevice = auto()
    cudaErrorFileNotFound = auto()
    cudaErrorFunctionNotLoaded = auto()
    cudaErrorGraphExecUpdateFailure = auto()
    cudaErrorHardwareStackError = auto()
    cudaErrorHostMemoryAlreadyRegistered = auto()
    cudaErrorHostMemoryNotRegistered = auto()
    cudaErrorIllegalAddress = auto()
    cudaErrorIllegalInstruction = auto()
    cudaErrorIllegalState = auto()
    cudaErrorIncompatibleDriverContext = auto()
    cudaErrorInitializationError = auto()
    cudaErrorInsufficientDriver = auto()
    cudaErrorInvalidAddressSpace = auto()
    cudaErrorInvalidChannelDescriptor = auto()
    cudaErrorInvalidClusterSize = auto()
    cudaErrorInvalidConfiguration = auto()
    cudaErrorInvalidDevice = auto()
    cudaErrorInvalidDeviceFunction = auto()
    cudaErrorInvalidDevicePointer = auto()
    cudaErrorInvalidFilterSetting = auto()
    cudaErrorInvalidGraphicsContext = auto()
    cudaErrorInvalidHostPointer = auto()
    cudaErrorInvalidKernelImage = auto()
    cudaErrorInvalidMemcpyDirection = auto()
    cudaErrorInvalidNormSetting = auto()
    cudaErrorInvalidPc = auto()
    cudaErrorInvalidPitchValue = auto()
    cudaErrorInvalidPtx = auto()
    cudaErrorInvalidResourceConfiguration = auto()
    cudaErrorInvalidResourceHandle = auto()
    cudaErrorInvalidResourceType = auto()
    cudaErrorInvalidSource = auto()
    cudaErrorInvalidSurface = auto()
    cudaErrorInvalidSymbol = auto()
    cudaErrorInvalidTexture = auto()
    cudaErrorInvalidTextureBinding = auto()
    cudaErrorInvalidValue = auto()
    cudaErrorJitCompilationDisabled = auto()
    cudaErrorJitCompilerNotFound = auto()
    cudaErrorLaunchFailure = auto()
    cudaErrorLaunchFileScopedSurf = auto()
    cudaErrorLaunchFileScopedTex = auto()
    cudaErrorLaunchIncompatibleTexturing = auto()
    cudaErrorLaunchMaxDepthExceeded = auto()
    cudaErrorLaunchOutOfResources = auto()
    cudaErrorLaunchPendingCountExceeded = auto()
    cudaErrorLaunchTimeout = auto()
    cudaErrorLossyQuery = auto()
    cudaErrorMapBufferObjectFailed = auto()
    cudaErrorMemoryAllocation = auto()
    cudaErrorMemoryValueTooLarge = auto()
    cudaErrorMisalignedAddress = auto()
    cudaErrorMissingConfiguration = auto()
    cudaErrorMixedDeviceExecution = auto()
    cudaErrorMpsClientTerminated = auto()
    cudaErrorMpsConnectionFailed = auto()
    cudaErrorMpsMaxClientsReached = auto()
    cudaErrorMpsMaxConnectionsReached = auto()
    cudaErrorMpsRpcFailure = auto()
    cudaErrorMpsServerNotReady = auto()
    cudaErrorNoDevice = auto()
    cudaErrorNoKernelImageForDevice = auto()
    cudaErrorNotMapped = auto()
    cudaErrorNotMappedAsArray = auto()
    cudaErrorNotMappedAsPointer = auto()
    cudaErrorNotPermitted = auto()
    cudaErrorNotReady = auto()
    cudaErrorNotSupported = auto()
    cudaErrorNotYetImplemented = auto()
    cudaErrorNvlinkUncorrectable = auto()
    cudaErrorOperatingSystem = auto()
    cudaErrorPeerAccessAlreadyEnabled = auto()
    cudaErrorPeerAccessNotEnabled = auto()
    cudaErrorPeerAccessUnsupported = auto()
    cudaErrorPriorLaunchFailure = auto()
    cudaErrorProfilerAlreadyStarted = auto()
    cudaErrorProfilerAlreadyStopped = auto()
    cudaErrorProfilerDisabled = auto()
    cudaErrorProfilerNotInitialized = auto()
    cudaErrorSetOnActiveProcess = auto()
    cudaErrorSharedObjectInitFailed = auto()
    cudaErrorSharedObjectSymbolNotFound = auto()
    cudaErrorSoftwareValidityNotEstablished = auto()
    cudaErrorStartupFailure = auto()
    cudaErrorStreamCaptureImplicit = auto()
    cudaErrorStreamCaptureInvalidated = auto()
    cudaErrorStreamCaptureIsolation = auto()
    cudaErrorStreamCaptureMerge = auto()
    cudaErrorStreamCaptureUnjoined = auto()
    cudaErrorStreamCaptureUnmatched = auto()
    cudaErrorStreamCaptureUnsupported = auto()
    cudaErrorStreamCaptureWrongThread = auto()
    cudaErrorStubLibrary = auto()
    cudaErrorSymbolNotFound = auto()
    cudaErrorSyncDepthExceeded = auto()
    cudaErrorSynchronizationError = auto()
    cudaErrorSystemDriverMismatch = auto()
    cudaErrorSystemNotReady = auto()
    cudaErrorTensorMemoryLeak = auto()
    cudaErrorTextureFetchFailed = auto()
    cudaErrorTextureNotBound = auto()
    cudaErrorTimeout = auto()
    cudaErrorTooManyPeers = auto()
    cudaErrorUnknown = auto()
    cudaErrorUnmapBufferObjectFailed = auto()
    cudaErrorUnsupportedDevSideSync = auto()
    cudaErrorUnsupportedExecAffinity = auto()
    cudaErrorUnsupportedLimit = auto()
    cudaErrorUnsupportedPtxVersion = auto()
    cudaSuccess = auto()

class cudaUUID_t:
    bytes: bytes
    def getPtr(self) -> int: ...

class cudaDeviceProp:
    ECCEnabled: int
    accessPolicyMaxWindowSize: int
    asyncEngineCount: int
    canMapHostMemory: int
    canUseHostPointerForRegisteredMem: int
    clockRate: int
    clusterLaunch: int
    computeMode: int
    computePreemptionSupported: int
    concurrentKernels: int
    concurrentManagedAccess: int
    cooperativeLaunch: int
    cooperativeMultiDeviceLaunch: int
    deferredMappingCudaArraySupported: int
    deviceOverlap: int
    directManagedMemAccessFromHost: int
    globalL1CacheSupported: int
    gpuDirectRDMAFlushWritesOptions: int
    gpuDirectRDMASupported: int
    gpuDirectRDMAWritesOrdering: int
    hostNativeAtomicSupported: int
    hostRegisterReadOnlySupported: int
    hostRegisterSupported: int
    integrated: int
    ipcEventSupported: int
    isMultiGpuBoard: int
    kernelExecTimeoutEnabled: int
    l2CacheSize: int
    localL1CacheSupported: int
    luid: bytes
    luidDeviceNodeMask: int
    major: int
    managedMemory: int
    maxBlocksPerMultiProcessor: int
    maxGridSize: list[int]
    maxSurface1D: int
    maxSurface1DLayered: list[int]
    maxSurface2D: list[int]
    maxSurface2DLayered: list[int]
    maxSurface3D: list[int]
    maxSurfaceCubemap: int
    maxSurfaceCubemapLayered: list[int]
    maxTexture1D: int
    maxTexture1DLayered: list[int]
    maxTexture1DLinear: list[int]
    maxTexture1DMipmap: list[int]
    maxTexture2D: list[int]
    maxTexture2DGather: list[int]
    maxTexture2DLayered: list[int]
    maxTexture2DLinear: list[int]
    maxTexture2DMipmap: list[int]
    maxTexture3D: list[int]
    maxTexture3DAlt: list[int]
    maxTextureCubemap: int
    maxTextureCubemapLayered: list[int]
    maxThreadsDim: list[int]
    maxThreadsPerBlock: int
    maxThreadsPerMultiProcessor: int
    memPitch: int
    memoryBusWidth: int
    memoryClockRate: int
    memoryPoolSupportedHandleTypes: int
    memoryPoolsSupported: int
    minor: int
    multiGpuBoardGroupID: int
    multiProcessorCount: int
    name: bytes
    pageableMemoryAccess: int
    pageableMemoryAccessUsesHostPageTables: int
    pciBusID: int
    pciDeviceID: int
    pciDomainID: int
    persistingL2CacheMaxSize: int
    regsPerBlock: int
    regsPerMultiprocessor: int
    reserved: list[int]
    reservedSharedMemPerBlock: int
    sharedMemPerBlock: int
    sharedMemPerBlockOptin: int
    sharedMemPerMultiprocessor: int
    singleToDoublePrecisionPerfRatio: int
    sparseCudaArraySupported: int
    streamPrioritiesSupported: int
    surfaceAlignment: int
    tccDriver: int
    textureAlignment: int
    texturePitchAlignment: int
    timelineSemaphoreInteropSupported: int
    totalConstMem: int
    totalGlobalMem: int
    unifiedAddressing: int
    unifiedFunctionPointers: int
    uuid: cudaUUID_t
    warpSize: int

    def __init__(self, void_ptr_ptr: int = 0) -> None: ...
    def getPtr(self) -> int: ...

class cudaStream_t:
    def getPtr(self) -> int: ...
    def __int__(self) -> int: ...

class cudaMemcpyKind(IntEnum):
    cudaMemcpyHostToHost = auto()
    cudaMemcpyHostToDevice = auto()
    cudaMemcpyDeviceToHost = auto()
    cudaMemcpyDeviceToDevice = auto()
    cudaMemcpyDefault = auto()

def cudaGetDevice() -> tuple[cudaError_t, int]: ...
def cudaSetDevice(device: int) -> tuple[cudaError_t]: ...
def cudaStreamCreate() -> tuple[cudaError_t, cudaStream_t]: ...
def cudaGetDeviceProperties(device: int) -> tuple[cudaError_t, cudaDeviceProp]: ...
def cudaGetDeviceCount() -> tuple[cudaError_t, int]: ...
def cudaMalloc(size: int) -> tuple[cudaError_t, int]: ...
def cudaFree(devPtr: int) -> tuple[cudaError_t]: ...
def cudaMallocAsync(size: int, stream: cudaStream_t) -> tuple[cudaError_t, int]: ...
def cudaFreeAsync(devPtr: int, stream: cudaStream_t) -> tuple[cudaError_t]: ...
def cudaStreamDestroy(stream: cudaStream_t) -> tuple[cudaError_t]: ...
def cudaMemcpyAsync(
    dst: int, src: int, size: int, kind: cudaMemcpyKind, stream: cudaStream_t
) -> tuple[cudaError_t]: ...
def cudaDeviceSynchronize() -> tuple[cudaError_t]: ...
def cudaStreamSynchronize(stream: cudaStream_t) -> tuple[cudaError_t]: ...
