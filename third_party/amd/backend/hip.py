# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-D__HIP_PLATFORM_AMD__', '-I/opt/rocm/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


class AsDictMixin:

    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith("PADDING_"):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, "_fields_"):
            return (f[0] for f in cls._fields_ if not f[0].startswith("PADDING"))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = type_((lambda callback: lambda *args: callback(*args))(bound_fields[name]))
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError("Cannot bind the following unknown callback(s) {}.{}".format(
                cls.__name__, bound_fields.keys()))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass


c_int128 = ctypes.c_ubyte * 16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte * 16


class FunctionFactoryStub:

    def __getattr__(self, _):
        return ctypes.CFUNCTYPE(lambda y: y)


_libraries = {}


def string_cast(char_pointer, encoding="utf-8", errors="strict"):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding="utf-8"):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))


_libraries["libamdhip64.so"] = ctypes.cdll.LoadLibrary("libamdhip64.so")

c__Ea_HIP_SUCCESS__enumvalues = {
    0: "HIP_SUCCESS",
    1: "HIP_ERROR_INVALID_VALUE",
    2: "HIP_ERROR_NOT_INITIALIZED",
    3: "HIP_ERROR_LAUNCH_OUT_OF_RESOURCES",
}
HIP_SUCCESS = 0
HIP_ERROR_INVALID_VALUE = 1
HIP_ERROR_NOT_INITIALIZED = 2
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = 3
c__Ea_HIP_SUCCESS = ctypes.c_uint32  # enum


class struct_c__SA_hipDeviceArch_t(Structure):
    pass


struct_c__SA_hipDeviceArch_t._pack_ = 1  # source:False
struct_c__SA_hipDeviceArch_t._fields_ = [
    ("hasGlobalInt32Atomics", ctypes.c_uint32, 1),
    ("hasGlobalFloatAtomicExch", ctypes.c_uint32, 1),
    ("hasSharedInt32Atomics", ctypes.c_uint32, 1),
    ("hasSharedFloatAtomicExch", ctypes.c_uint32, 1),
    ("hasFloatAtomicAdd", ctypes.c_uint32, 1),
    ("hasGlobalInt64Atomics", ctypes.c_uint32, 1),
    ("hasSharedInt64Atomics", ctypes.c_uint32, 1),
    ("hasDoubles", ctypes.c_uint32, 1),
    ("hasWarpVote", ctypes.c_uint32, 1),
    ("hasWarpBallot", ctypes.c_uint32, 1),
    ("hasWarpShuffle", ctypes.c_uint32, 1),
    ("hasFunnelShift", ctypes.c_uint32, 1),
    ("hasThreadFenceSystem", ctypes.c_uint32, 1),
    ("hasSyncThreadsExt", ctypes.c_uint32, 1),
    ("hasSurfaceFuncs", ctypes.c_uint32, 1),
    ("has3dGrid", ctypes.c_uint32, 1),
    ("hasDynamicParallelism", ctypes.c_uint32, 1),
    ("PADDING_0", ctypes.c_uint16, 15),
]

hipDeviceArch_t = struct_c__SA_hipDeviceArch_t


class struct_hipUUID_t(Structure):
    pass


struct_hipUUID_t._pack_ = 1  # source:False
struct_hipUUID_t._fields_ = [
    ("bytes", ctypes.c_char * 16),
]

hipUUID = struct_hipUUID_t


class struct_hipDeviceProp_tR0600(Structure):
    pass


struct_hipDeviceProp_tR0600._pack_ = 1  # source:False
struct_hipDeviceProp_tR0600._fields_ = [
    ("name", ctypes.c_char * 256),
    ("uuid", hipUUID),
    ("luid", ctypes.c_char * 8),
    ("luidDeviceNodeMask", ctypes.c_uint32),
    ("PADDING_0", ctypes.c_ubyte * 4),
    ("totalGlobalMem", ctypes.c_uint64),
    ("sharedMemPerBlock", ctypes.c_uint64),
    ("regsPerBlock", ctypes.c_int32),
    ("warpSize", ctypes.c_int32),
    ("memPitch", ctypes.c_uint64),
    ("maxThreadsPerBlock", ctypes.c_int32),
    ("maxThreadsDim", ctypes.c_int32 * 3),
    ("maxGridSize", ctypes.c_int32 * 3),
    ("clockRate", ctypes.c_int32),
    ("totalConstMem", ctypes.c_uint64),
    ("major", ctypes.c_int32),
    ("minor", ctypes.c_int32),
    ("textureAlignment", ctypes.c_uint64),
    ("texturePitchAlignment", ctypes.c_uint64),
    ("deviceOverlap", ctypes.c_int32),
    ("multiProcessorCount", ctypes.c_int32),
    ("kernelExecTimeoutEnabled", ctypes.c_int32),
    ("integrated", ctypes.c_int32),
    ("canMapHostMemory", ctypes.c_int32),
    ("computeMode", ctypes.c_int32),
    ("maxTexture1D", ctypes.c_int32),
    ("maxTexture1DMipmap", ctypes.c_int32),
    ("maxTexture1DLinear", ctypes.c_int32),
    ("maxTexture2D", ctypes.c_int32 * 2),
    ("maxTexture2DMipmap", ctypes.c_int32 * 2),
    ("maxTexture2DLinear", ctypes.c_int32 * 3),
    ("maxTexture2DGather", ctypes.c_int32 * 2),
    ("maxTexture3D", ctypes.c_int32 * 3),
    ("maxTexture3DAlt", ctypes.c_int32 * 3),
    ("maxTextureCubemap", ctypes.c_int32),
    ("maxTexture1DLayered", ctypes.c_int32 * 2),
    ("maxTexture2DLayered", ctypes.c_int32 * 3),
    ("maxTextureCubemapLayered", ctypes.c_int32 * 2),
    ("maxSurface1D", ctypes.c_int32),
    ("maxSurface2D", ctypes.c_int32 * 2),
    ("maxSurface3D", ctypes.c_int32 * 3),
    ("maxSurface1DLayered", ctypes.c_int32 * 2),
    ("maxSurface2DLayered", ctypes.c_int32 * 3),
    ("maxSurfaceCubemap", ctypes.c_int32),
    ("maxSurfaceCubemapLayered", ctypes.c_int32 * 2),
    ("surfaceAlignment", ctypes.c_uint64),
    ("concurrentKernels", ctypes.c_int32),
    ("ECCEnabled", ctypes.c_int32),
    ("pciBusID", ctypes.c_int32),
    ("pciDeviceID", ctypes.c_int32),
    ("pciDomainID", ctypes.c_int32),
    ("tccDriver", ctypes.c_int32),
    ("asyncEngineCount", ctypes.c_int32),
    ("unifiedAddressing", ctypes.c_int32),
    ("memoryClockRate", ctypes.c_int32),
    ("memoryBusWidth", ctypes.c_int32),
    ("l2CacheSize", ctypes.c_int32),
    ("persistingL2CacheMaxSize", ctypes.c_int32),
    ("maxThreadsPerMultiProcessor", ctypes.c_int32),
    ("streamPrioritiesSupported", ctypes.c_int32),
    ("globalL1CacheSupported", ctypes.c_int32),
    ("localL1CacheSupported", ctypes.c_int32),
    ("sharedMemPerMultiprocessor", ctypes.c_uint64),
    ("regsPerMultiprocessor", ctypes.c_int32),
    ("managedMemory", ctypes.c_int32),
    ("isMultiGpuBoard", ctypes.c_int32),
    ("multiGpuBoardGroupID", ctypes.c_int32),
    ("hostNativeAtomicSupported", ctypes.c_int32),
    ("singleToDoublePrecisionPerfRatio", ctypes.c_int32),
    ("pageableMemoryAccess", ctypes.c_int32),
    ("concurrentManagedAccess", ctypes.c_int32),
    ("computePreemptionSupported", ctypes.c_int32),
    ("canUseHostPointerForRegisteredMem", ctypes.c_int32),
    ("cooperativeLaunch", ctypes.c_int32),
    ("cooperativeMultiDeviceLaunch", ctypes.c_int32),
    ("sharedMemPerBlockOptin", ctypes.c_uint64),
    ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int32),
    ("directManagedMemAccessFromHost", ctypes.c_int32),
    ("maxBlocksPerMultiProcessor", ctypes.c_int32),
    ("accessPolicyMaxWindowSize", ctypes.c_int32),
    ("reservedSharedMemPerBlock", ctypes.c_uint64),
    ("hostRegisterSupported", ctypes.c_int32),
    ("sparseHipArraySupported", ctypes.c_int32),
    ("hostRegisterReadOnlySupported", ctypes.c_int32),
    ("timelineSemaphoreInteropSupported", ctypes.c_int32),
    ("memoryPoolsSupported", ctypes.c_int32),
    ("gpuDirectRDMASupported", ctypes.c_int32),
    ("gpuDirectRDMAFlushWritesOptions", ctypes.c_uint32),
    ("gpuDirectRDMAWritesOrdering", ctypes.c_int32),
    ("memoryPoolSupportedHandleTypes", ctypes.c_uint32),
    ("deferredMappingHipArraySupported", ctypes.c_int32),
    ("ipcEventSupported", ctypes.c_int32),
    ("clusterLaunch", ctypes.c_int32),
    ("unifiedFunctionPointers", ctypes.c_int32),
    ("reserved", ctypes.c_int32 * 63),
    ("hipReserved", ctypes.c_int32 * 32),
    ("gcnArchName", ctypes.c_char * 256),
    ("maxSharedMemoryPerMultiProcessor", ctypes.c_uint64),
    ("clockInstructionRate", ctypes.c_int32),
    ("arch", hipDeviceArch_t),
    ("hdpMemFlushCntl", ctypes.POINTER(ctypes.c_uint32)),
    ("hdpRegFlushCntl", ctypes.POINTER(ctypes.c_uint32)),
    ("cooperativeMultiDeviceUnmatchedFunc", ctypes.c_int32),
    ("cooperativeMultiDeviceUnmatchedGridDim", ctypes.c_int32),
    ("cooperativeMultiDeviceUnmatchedBlockDim", ctypes.c_int32),
    ("cooperativeMultiDeviceUnmatchedSharedMem", ctypes.c_int32),
    ("isLargeBar", ctypes.c_int32),
    ("asicRevision", ctypes.c_int32),
]

hipDeviceProp_tR0600 = struct_hipDeviceProp_tR0600

hipMemoryType__enumvalues = {
    0: "hipMemoryTypeUnregistered",
    1: "hipMemoryTypeHost",
    2: "hipMemoryTypeDevice",
    3: "hipMemoryTypeManaged",
    10: "hipMemoryTypeArray",
    11: "hipMemoryTypeUnified",
}
hipMemoryTypeUnregistered = 0
hipMemoryTypeHost = 1
hipMemoryTypeDevice = 2
hipMemoryTypeManaged = 3
hipMemoryTypeArray = 10
hipMemoryTypeUnified = 11
hipMemoryType = ctypes.c_uint32  # enum


class struct_hipPointerAttribute_t(Structure):
    pass


struct_hipPointerAttribute_t._pack_ = 1  # source:False
struct_hipPointerAttribute_t._fields_ = [
    ("type", hipMemoryType),
    ("device", ctypes.c_int32),
    ("devicePointer", ctypes.POINTER(None)),
    ("hostPointer", ctypes.POINTER(None)),
    ("isManaged", ctypes.c_int32),
    ("allocationFlags", ctypes.c_uint32),
]

hipPointerAttribute_t = struct_hipPointerAttribute_t

hipError_t__enumvalues = {
    0: "hipSuccess",
    1: "hipErrorInvalidValue",
    2: "hipErrorOutOfMemory",
    2: "hipErrorMemoryAllocation",
    3: "hipErrorNotInitialized",
    3: "hipErrorInitializationError",
    4: "hipErrorDeinitialized",
    5: "hipErrorProfilerDisabled",
    6: "hipErrorProfilerNotInitialized",
    7: "hipErrorProfilerAlreadyStarted",
    8: "hipErrorProfilerAlreadyStopped",
    9: "hipErrorInvalidConfiguration",
    12: "hipErrorInvalidPitchValue",
    13: "hipErrorInvalidSymbol",
    17: "hipErrorInvalidDevicePointer",
    21: "hipErrorInvalidMemcpyDirection",
    35: "hipErrorInsufficientDriver",
    52: "hipErrorMissingConfiguration",
    53: "hipErrorPriorLaunchFailure",
    98: "hipErrorInvalidDeviceFunction",
    100: "hipErrorNoDevice",
    101: "hipErrorInvalidDevice",
    200: "hipErrorInvalidImage",
    201: "hipErrorInvalidContext",
    202: "hipErrorContextAlreadyCurrent",
    205: "hipErrorMapFailed",
    205: "hipErrorMapBufferObjectFailed",
    206: "hipErrorUnmapFailed",
    207: "hipErrorArrayIsMapped",
    208: "hipErrorAlreadyMapped",
    209: "hipErrorNoBinaryForGpu",
    210: "hipErrorAlreadyAcquired",
    211: "hipErrorNotMapped",
    212: "hipErrorNotMappedAsArray",
    213: "hipErrorNotMappedAsPointer",
    214: "hipErrorECCNotCorrectable",
    215: "hipErrorUnsupportedLimit",
    216: "hipErrorContextAlreadyInUse",
    217: "hipErrorPeerAccessUnsupported",
    218: "hipErrorInvalidKernelFile",
    219: "hipErrorInvalidGraphicsContext",
    300: "hipErrorInvalidSource",
    301: "hipErrorFileNotFound",
    302: "hipErrorSharedObjectSymbolNotFound",
    303: "hipErrorSharedObjectInitFailed",
    304: "hipErrorOperatingSystem",
    400: "hipErrorInvalidHandle",
    400: "hipErrorInvalidResourceHandle",
    401: "hipErrorIllegalState",
    500: "hipErrorNotFound",
    600: "hipErrorNotReady",
    700: "hipErrorIllegalAddress",
    701: "hipErrorLaunchOutOfResources",
    702: "hipErrorLaunchTimeOut",
    704: "hipErrorPeerAccessAlreadyEnabled",
    705: "hipErrorPeerAccessNotEnabled",
    708: "hipErrorSetOnActiveProcess",
    709: "hipErrorContextIsDestroyed",
    710: "hipErrorAssert",
    712: "hipErrorHostMemoryAlreadyRegistered",
    713: "hipErrorHostMemoryNotRegistered",
    719: "hipErrorLaunchFailure",
    720: "hipErrorCooperativeLaunchTooLarge",
    801: "hipErrorNotSupported",
    900: "hipErrorStreamCaptureUnsupported",
    901: "hipErrorStreamCaptureInvalidated",
    902: "hipErrorStreamCaptureMerge",
    903: "hipErrorStreamCaptureUnmatched",
    904: "hipErrorStreamCaptureUnjoined",
    905: "hipErrorStreamCaptureIsolation",
    906: "hipErrorStreamCaptureImplicit",
    907: "hipErrorCapturedEvent",
    908: "hipErrorStreamCaptureWrongThread",
    910: "hipErrorGraphExecUpdateFailure",
    999: "hipErrorUnknown",
    1052: "hipErrorRuntimeMemory",
    1053: "hipErrorRuntimeOther",
    1054: "hipErrorTbd",
}
hipSuccess = 0
hipErrorInvalidValue = 1
hipErrorOutOfMemory = 2
hipErrorMemoryAllocation = 2
hipErrorNotInitialized = 3
hipErrorInitializationError = 3
hipErrorDeinitialized = 4
hipErrorProfilerDisabled = 5
hipErrorProfilerNotInitialized = 6
hipErrorProfilerAlreadyStarted = 7
hipErrorProfilerAlreadyStopped = 8
hipErrorInvalidConfiguration = 9
hipErrorInvalidPitchValue = 12
hipErrorInvalidSymbol = 13
hipErrorInvalidDevicePointer = 17
hipErrorInvalidMemcpyDirection = 21
hipErrorInsufficientDriver = 35
hipErrorMissingConfiguration = 52
hipErrorPriorLaunchFailure = 53
hipErrorInvalidDeviceFunction = 98
hipErrorNoDevice = 100
hipErrorInvalidDevice = 101
hipErrorInvalidImage = 200
hipErrorInvalidContext = 201
hipErrorContextAlreadyCurrent = 202
hipErrorMapFailed = 205
hipErrorMapBufferObjectFailed = 205
hipErrorUnmapFailed = 206
hipErrorArrayIsMapped = 207
hipErrorAlreadyMapped = 208
hipErrorNoBinaryForGpu = 209
hipErrorAlreadyAcquired = 210
hipErrorNotMapped = 211
hipErrorNotMappedAsArray = 212
hipErrorNotMappedAsPointer = 213
hipErrorECCNotCorrectable = 214
hipErrorUnsupportedLimit = 215
hipErrorContextAlreadyInUse = 216
hipErrorPeerAccessUnsupported = 217
hipErrorInvalidKernelFile = 218
hipErrorInvalidGraphicsContext = 219
hipErrorInvalidSource = 300
hipErrorFileNotFound = 301
hipErrorSharedObjectSymbolNotFound = 302
hipErrorSharedObjectInitFailed = 303
hipErrorOperatingSystem = 304
hipErrorInvalidHandle = 400
hipErrorInvalidResourceHandle = 400
hipErrorIllegalState = 401
hipErrorNotFound = 500
hipErrorNotReady = 600
hipErrorIllegalAddress = 700
hipErrorLaunchOutOfResources = 701
hipErrorLaunchTimeOut = 702
hipErrorPeerAccessAlreadyEnabled = 704
hipErrorPeerAccessNotEnabled = 705
hipErrorSetOnActiveProcess = 708
hipErrorContextIsDestroyed = 709
hipErrorAssert = 710
hipErrorHostMemoryAlreadyRegistered = 712
hipErrorHostMemoryNotRegistered = 713
hipErrorLaunchFailure = 719
hipErrorCooperativeLaunchTooLarge = 720
hipErrorNotSupported = 801
hipErrorStreamCaptureUnsupported = 900
hipErrorStreamCaptureInvalidated = 901
hipErrorStreamCaptureMerge = 902
hipErrorStreamCaptureUnmatched = 903
hipErrorStreamCaptureUnjoined = 904
hipErrorStreamCaptureIsolation = 905
hipErrorStreamCaptureImplicit = 906
hipErrorCapturedEvent = 907
hipErrorStreamCaptureWrongThread = 908
hipErrorGraphExecUpdateFailure = 910
hipErrorUnknown = 999
hipErrorRuntimeMemory = 1052
hipErrorRuntimeOther = 1053
hipErrorTbd = 1054
hipError_t = ctypes.c_uint32  # enum

hipDeviceAttribute_t__enumvalues = {
    0: "hipDeviceAttributeCudaCompatibleBegin",
    0: "hipDeviceAttributeEccEnabled",
    1: "hipDeviceAttributeAccessPolicyMaxWindowSize",
    2: "hipDeviceAttributeAsyncEngineCount",
    3: "hipDeviceAttributeCanMapHostMemory",
    4: "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
    5: "hipDeviceAttributeClockRate",
    6: "hipDeviceAttributeComputeMode",
    7: "hipDeviceAttributeComputePreemptionSupported",
    8: "hipDeviceAttributeConcurrentKernels",
    9: "hipDeviceAttributeConcurrentManagedAccess",
    10: "hipDeviceAttributeCooperativeLaunch",
    11: "hipDeviceAttributeCooperativeMultiDeviceLaunch",
    12: "hipDeviceAttributeDeviceOverlap",
    13: "hipDeviceAttributeDirectManagedMemAccessFromHost",
    14: "hipDeviceAttributeGlobalL1CacheSupported",
    15: "hipDeviceAttributeHostNativeAtomicSupported",
    16: "hipDeviceAttributeIntegrated",
    17: "hipDeviceAttributeIsMultiGpuBoard",
    18: "hipDeviceAttributeKernelExecTimeout",
    19: "hipDeviceAttributeL2CacheSize",
    20: "hipDeviceAttributeLocalL1CacheSupported",
    21: "hipDeviceAttributeLuid",
    22: "hipDeviceAttributeLuidDeviceNodeMask",
    23: "hipDeviceAttributeComputeCapabilityMajor",
    24: "hipDeviceAttributeManagedMemory",
    25: "hipDeviceAttributeMaxBlocksPerMultiProcessor",
    26: "hipDeviceAttributeMaxBlockDimX",
    27: "hipDeviceAttributeMaxBlockDimY",
    28: "hipDeviceAttributeMaxBlockDimZ",
    29: "hipDeviceAttributeMaxGridDimX",
    30: "hipDeviceAttributeMaxGridDimY",
    31: "hipDeviceAttributeMaxGridDimZ",
    32: "hipDeviceAttributeMaxSurface1D",
    33: "hipDeviceAttributeMaxSurface1DLayered",
    34: "hipDeviceAttributeMaxSurface2D",
    35: "hipDeviceAttributeMaxSurface2DLayered",
    36: "hipDeviceAttributeMaxSurface3D",
    37: "hipDeviceAttributeMaxSurfaceCubemap",
    38: "hipDeviceAttributeMaxSurfaceCubemapLayered",
    39: "hipDeviceAttributeMaxTexture1DWidth",
    40: "hipDeviceAttributeMaxTexture1DLayered",
    41: "hipDeviceAttributeMaxTexture1DLinear",
    42: "hipDeviceAttributeMaxTexture1DMipmap",
    43: "hipDeviceAttributeMaxTexture2DWidth",
    44: "hipDeviceAttributeMaxTexture2DHeight",
    45: "hipDeviceAttributeMaxTexture2DGather",
    46: "hipDeviceAttributeMaxTexture2DLayered",
    47: "hipDeviceAttributeMaxTexture2DLinear",
    48: "hipDeviceAttributeMaxTexture2DMipmap",
    49: "hipDeviceAttributeMaxTexture3DWidth",
    50: "hipDeviceAttributeMaxTexture3DHeight",
    51: "hipDeviceAttributeMaxTexture3DDepth",
    52: "hipDeviceAttributeMaxTexture3DAlt",
    53: "hipDeviceAttributeMaxTextureCubemap",
    54: "hipDeviceAttributeMaxTextureCubemapLayered",
    55: "hipDeviceAttributeMaxThreadsDim",
    56: "hipDeviceAttributeMaxThreadsPerBlock",
    57: "hipDeviceAttributeMaxThreadsPerMultiProcessor",
    58: "hipDeviceAttributeMaxPitch",
    59: "hipDeviceAttributeMemoryBusWidth",
    60: "hipDeviceAttributeMemoryClockRate",
    61: "hipDeviceAttributeComputeCapabilityMinor",
    62: "hipDeviceAttributeMultiGpuBoardGroupID",
    63: "hipDeviceAttributeMultiprocessorCount",
    64: "hipDeviceAttributeUnused1",
    65: "hipDeviceAttributePageableMemoryAccess",
    66: "hipDeviceAttributePageableMemoryAccessUsesHostPageTables",
    67: "hipDeviceAttributePciBusId",
    68: "hipDeviceAttributePciDeviceId",
    69: "hipDeviceAttributePciDomainID",
    70: "hipDeviceAttributePersistingL2CacheMaxSize",
    71: "hipDeviceAttributeMaxRegistersPerBlock",
    72: "hipDeviceAttributeMaxRegistersPerMultiprocessor",
    73: "hipDeviceAttributeReservedSharedMemPerBlock",
    74: "hipDeviceAttributeMaxSharedMemoryPerBlock",
    75: "hipDeviceAttributeSharedMemPerBlockOptin",
    76: "hipDeviceAttributeSharedMemPerMultiprocessor",
    77: "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
    78: "hipDeviceAttributeStreamPrioritiesSupported",
    79: "hipDeviceAttributeSurfaceAlignment",
    80: "hipDeviceAttributeTccDriver",
    81: "hipDeviceAttributeTextureAlignment",
    82: "hipDeviceAttributeTexturePitchAlignment",
    83: "hipDeviceAttributeTotalConstantMemory",
    84: "hipDeviceAttributeTotalGlobalMem",
    85: "hipDeviceAttributeUnifiedAddressing",
    86: "hipDeviceAttributeUnused2",
    87: "hipDeviceAttributeWarpSize",
    88: "hipDeviceAttributeMemoryPoolsSupported",
    89: "hipDeviceAttributeVirtualMemoryManagementSupported",
    90: "hipDeviceAttributeHostRegisterSupported",
    9999: "hipDeviceAttributeCudaCompatibleEnd",
    10000: "hipDeviceAttributeAmdSpecificBegin",
    10000: "hipDeviceAttributeClockInstructionRate",
    10001: "hipDeviceAttributeUnused3",
    10002: "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
    10003: "hipDeviceAttributeUnused4",
    10004: "hipDeviceAttributeUnused5",
    10005: "hipDeviceAttributeHdpMemFlushCntl",
    10006: "hipDeviceAttributeHdpRegFlushCntl",
    10007: "hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc",
    10008: "hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim",
    10009: "hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim",
    10010: "hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem",
    10011: "hipDeviceAttributeIsLargeBar",
    10012: "hipDeviceAttributeAsicRevision",
    10013: "hipDeviceAttributeCanUseStreamWaitValue",
    10014: "hipDeviceAttributeImageSupport",
    10015: "hipDeviceAttributePhysicalMultiProcessorCount",
    10016: "hipDeviceAttributeFineGrainSupport",
    10017: "hipDeviceAttributeWallClockRate",
    19999: "hipDeviceAttributeAmdSpecificEnd",
    20000: "hipDeviceAttributeVendorSpecificBegin",
}
hipDeviceAttributeCudaCompatibleBegin = 0
hipDeviceAttributeEccEnabled = 0
hipDeviceAttributeAccessPolicyMaxWindowSize = 1
hipDeviceAttributeAsyncEngineCount = 2
hipDeviceAttributeCanMapHostMemory = 3
hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4
hipDeviceAttributeClockRate = 5
hipDeviceAttributeComputeMode = 6
hipDeviceAttributeComputePreemptionSupported = 7
hipDeviceAttributeConcurrentKernels = 8
hipDeviceAttributeConcurrentManagedAccess = 9
hipDeviceAttributeCooperativeLaunch = 10
hipDeviceAttributeCooperativeMultiDeviceLaunch = 11
hipDeviceAttributeDeviceOverlap = 12
hipDeviceAttributeDirectManagedMemAccessFromHost = 13
hipDeviceAttributeGlobalL1CacheSupported = 14
hipDeviceAttributeHostNativeAtomicSupported = 15
hipDeviceAttributeIntegrated = 16
hipDeviceAttributeIsMultiGpuBoard = 17
hipDeviceAttributeKernelExecTimeout = 18
hipDeviceAttributeL2CacheSize = 19
hipDeviceAttributeLocalL1CacheSupported = 20
hipDeviceAttributeLuid = 21
hipDeviceAttributeLuidDeviceNodeMask = 22
hipDeviceAttributeComputeCapabilityMajor = 23
hipDeviceAttributeManagedMemory = 24
hipDeviceAttributeMaxBlocksPerMultiProcessor = 25
hipDeviceAttributeMaxBlockDimX = 26
hipDeviceAttributeMaxBlockDimY = 27
hipDeviceAttributeMaxBlockDimZ = 28
hipDeviceAttributeMaxGridDimX = 29
hipDeviceAttributeMaxGridDimY = 30
hipDeviceAttributeMaxGridDimZ = 31
hipDeviceAttributeMaxSurface1D = 32
hipDeviceAttributeMaxSurface1DLayered = 33
hipDeviceAttributeMaxSurface2D = 34
hipDeviceAttributeMaxSurface2DLayered = 35
hipDeviceAttributeMaxSurface3D = 36
hipDeviceAttributeMaxSurfaceCubemap = 37
hipDeviceAttributeMaxSurfaceCubemapLayered = 38
hipDeviceAttributeMaxTexture1DWidth = 39
hipDeviceAttributeMaxTexture1DLayered = 40
hipDeviceAttributeMaxTexture1DLinear = 41
hipDeviceAttributeMaxTexture1DMipmap = 42
hipDeviceAttributeMaxTexture2DWidth = 43
hipDeviceAttributeMaxTexture2DHeight = 44
hipDeviceAttributeMaxTexture2DGather = 45
hipDeviceAttributeMaxTexture2DLayered = 46
hipDeviceAttributeMaxTexture2DLinear = 47
hipDeviceAttributeMaxTexture2DMipmap = 48
hipDeviceAttributeMaxTexture3DWidth = 49
hipDeviceAttributeMaxTexture3DHeight = 50
hipDeviceAttributeMaxTexture3DDepth = 51
hipDeviceAttributeMaxTexture3DAlt = 52
hipDeviceAttributeMaxTextureCubemap = 53
hipDeviceAttributeMaxTextureCubemapLayered = 54
hipDeviceAttributeMaxThreadsDim = 55
hipDeviceAttributeMaxThreadsPerBlock = 56
hipDeviceAttributeMaxThreadsPerMultiProcessor = 57
hipDeviceAttributeMaxPitch = 58
hipDeviceAttributeMemoryBusWidth = 59
hipDeviceAttributeMemoryClockRate = 60
hipDeviceAttributeComputeCapabilityMinor = 61
hipDeviceAttributeMultiGpuBoardGroupID = 62
hipDeviceAttributeMultiprocessorCount = 63
hipDeviceAttributeUnused1 = 64
hipDeviceAttributePageableMemoryAccess = 65
hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66
hipDeviceAttributePciBusId = 67
hipDeviceAttributePciDeviceId = 68
hipDeviceAttributePciDomainID = 69
hipDeviceAttributePersistingL2CacheMaxSize = 70
hipDeviceAttributeMaxRegistersPerBlock = 71
hipDeviceAttributeMaxRegistersPerMultiprocessor = 72
hipDeviceAttributeReservedSharedMemPerBlock = 73
hipDeviceAttributeMaxSharedMemoryPerBlock = 74
hipDeviceAttributeSharedMemPerBlockOptin = 75
hipDeviceAttributeSharedMemPerMultiprocessor = 76
hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77
hipDeviceAttributeStreamPrioritiesSupported = 78
hipDeviceAttributeSurfaceAlignment = 79
hipDeviceAttributeTccDriver = 80
hipDeviceAttributeTextureAlignment = 81
hipDeviceAttributeTexturePitchAlignment = 82
hipDeviceAttributeTotalConstantMemory = 83
hipDeviceAttributeTotalGlobalMem = 84
hipDeviceAttributeUnifiedAddressing = 85
hipDeviceAttributeUnused2 = 86
hipDeviceAttributeWarpSize = 87
hipDeviceAttributeMemoryPoolsSupported = 88
hipDeviceAttributeVirtualMemoryManagementSupported = 89
hipDeviceAttributeHostRegisterSupported = 90
hipDeviceAttributeCudaCompatibleEnd = 9999
hipDeviceAttributeAmdSpecificBegin = 10000
hipDeviceAttributeClockInstructionRate = 10000
hipDeviceAttributeUnused3 = 10001
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002
hipDeviceAttributeUnused4 = 10003
hipDeviceAttributeUnused5 = 10004
hipDeviceAttributeHdpMemFlushCntl = 10005
hipDeviceAttributeHdpRegFlushCntl = 10006
hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007
hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008
hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009
hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010
hipDeviceAttributeIsLargeBar = 10011
hipDeviceAttributeAsicRevision = 10012
hipDeviceAttributeCanUseStreamWaitValue = 10013
hipDeviceAttributeImageSupport = 10014
hipDeviceAttributePhysicalMultiProcessorCount = 10015
hipDeviceAttributeFineGrainSupport = 10016
hipDeviceAttributeWallClockRate = 10017
hipDeviceAttributeAmdSpecificEnd = 19999
hipDeviceAttributeVendorSpecificBegin = 20000
hipDeviceAttribute_t = ctypes.c_uint32  # enum
hipDeviceptr_t = ctypes.POINTER(None)
hipFunction_attribute__enumvalues = {
    0: "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    1: "HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    2: "HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
    3: "HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
    4: "HIP_FUNC_ATTRIBUTE_NUM_REGS",
    5: "HIP_FUNC_ATTRIBUTE_PTX_VERSION",
    6: "HIP_FUNC_ATTRIBUTE_BINARY_VERSION",
    7: "HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA",
    8: "HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    9: "HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    10: "HIP_FUNC_ATTRIBUTE_MAX",
}
HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
HIP_FUNC_ATTRIBUTE_NUM_REGS = 4
HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5
HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6
HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
HIP_FUNC_ATTRIBUTE_MAX = 10
hipFunction_attribute = ctypes.c_uint32  # enum


class struct_ihipStream_t(Structure):
    pass


hipStream_t = ctypes.POINTER(struct_ihipStream_t)


class struct_ihipModule_t(Structure):
    pass


hipModule_t = ctypes.POINTER(struct_ihipModule_t)


class struct_ihipModuleSymbol_t(Structure):
    pass


hipFunction_t = ctypes.POINTER(struct_ihipModuleSymbol_t)
hipJitOption__enumvalues = {
    0: "hipJitOptionMaxRegisters",
    1: "hipJitOptionThreadsPerBlock",
    2: "hipJitOptionWallTime",
    3: "hipJitOptionInfoLogBuffer",
    4: "hipJitOptionInfoLogBufferSizeBytes",
    5: "hipJitOptionErrorLogBuffer",
    6: "hipJitOptionErrorLogBufferSizeBytes",
    7: "hipJitOptionOptimizationLevel",
    8: "hipJitOptionTargetFromContext",
    9: "hipJitOptionTarget",
    10: "hipJitOptionFallbackStrategy",
    11: "hipJitOptionGenerateDebugInfo",
    12: "hipJitOptionLogVerbose",
    13: "hipJitOptionGenerateLineInfo",
    14: "hipJitOptionCacheMode",
    15: "hipJitOptionSm3xOpt",
    16: "hipJitOptionFastCompile",
    17: "hipJitOptionNumOptions",
}
hipJitOptionMaxRegisters = 0
hipJitOptionThreadsPerBlock = 1
hipJitOptionWallTime = 2
hipJitOptionInfoLogBuffer = 3
hipJitOptionInfoLogBufferSizeBytes = 4
hipJitOptionErrorLogBuffer = 5
hipJitOptionErrorLogBufferSizeBytes = 6
hipJitOptionOptimizationLevel = 7
hipJitOptionTargetFromContext = 8
hipJitOptionTarget = 9
hipJitOptionFallbackStrategy = 10
hipJitOptionGenerateDebugInfo = 11
hipJitOptionLogVerbose = 12
hipJitOptionGenerateLineInfo = 13
hipJitOptionCacheMode = 14
hipJitOptionSm3xOpt = 15
hipJitOptionFastCompile = 16
hipJitOptionNumOptions = 17
hipJitOption = ctypes.c_uint32  # enum

hipGetDevicePropertiesR0600 = _libraries["libamdhip64.so"].hipGetDevicePropertiesR0600
hipGetDevicePropertiesR0600.restype = hipError_t
hipGetDevicePropertiesR0600.argtypes = [
    ctypes.POINTER(struct_hipDeviceProp_tR0600),
    ctypes.c_int32,
]

hipPointerGetAttributes = _libraries["libamdhip64.so"].hipPointerGetAttributes
hipPointerGetAttributes.restype = hipError_t
hipPointerGetAttributes.argtypes = [
    ctypes.POINTER(struct_hipPointerAttribute_t),
    ctypes.POINTER(None),
]

hipModuleGetFunction = _libraries["libamdhip64.so"].hipModuleGetFunction
hipModuleGetFunction.restype = hipError_t
hipModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.POINTER(struct_ihipModuleSymbol_t)),
    hipModule_t,
    ctypes.POINTER(ctypes.c_char),
]

hipFuncGetAttribute = _libraries["libamdhip64.so"].hipFuncGetAttribute
hipFuncGetAttribute.restype = hipError_t
hipFuncGetAttribute.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    hipFunction_attribute,
    hipFunction_t,
]

hipModuleLoadDataEx = _libraries["libamdhip64.so"].hipModuleLoadDataEx
hipModuleLoadDataEx.restype = hipError_t
hipModuleLoadDataEx.argtypes = [
    ctypes.POINTER(ctypes.POINTER(struct_ihipModule_t)),
    ctypes.POINTER(None),
    ctypes.c_uint32,
    ctypes.POINTER(hipJitOption),
    ctypes.POINTER(ctypes.POINTER(None)),
]

hipModuleLaunchKernel = _libraries["libamdhip64.so"].hipModuleLaunchKernel
hipModuleLaunchKernel.restype = hipError_t
hipModuleLaunchKernel.argtypes = [
    hipFunction_t,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    hipStream_t,
    ctypes.POINTER(ctypes.POINTER(None)),
    ctypes.POINTER(ctypes.POINTER(None)),
]

hipDeviceProp_t = hipDeviceProp_tR0600
hipGetDeviceProperties = hipGetDevicePropertiesR0600

hipGetErrorString = _libraries["libamdhip64.so"].hipGetErrorString
hipGetErrorString.restype = ctypes.POINTER(ctypes.c_char)
hipGetErrorString.argtypes = [hipError_t]

__all__ = [
    "HIP_ERROR_INVALID_VALUE",
    "HIP_ERROR_LAUNCH_OUT_OF_RESOURCES",
    "HIP_ERROR_NOT_INITIALIZED",
    "HIP_FUNC_ATTRIBUTE_BINARY_VERSION",
    "HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA",
    "HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
    "HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
    "HIP_FUNC_ATTRIBUTE_MAX",
    "HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    "HIP_FUNC_ATTRIBUTE_NUM_REGS",
    "HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    "HIP_FUNC_ATTRIBUTE_PTX_VERSION",
    "HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    "HIP_SUCCESS",
    "hipDeviceArch_t",
    "hipDeviceAttributeAccessPolicyMaxWindowSize",
    "hipDeviceAttributeAmdSpecificBegin",
    "hipDeviceAttributeAmdSpecificEnd",
    "hipDeviceAttributeAsicRevision",
    "hipDeviceAttributeAsyncEngineCount",
    "hipDeviceAttributeCanMapHostMemory",
    "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
    "hipDeviceAttributeCanUseStreamWaitValue",
    "hipDeviceAttributeClockInstructionRate",
    "hipDeviceAttributeClockRate",
    "hipDeviceAttributeComputeCapabilityMajor",
    "hipDeviceAttributeComputeCapabilityMinor",
    "hipDeviceAttributeComputeMode",
    "hipDeviceAttributeComputePreemptionSupported",
    "hipDeviceAttributeConcurrentKernels",
    "hipDeviceAttributeConcurrentManagedAccess",
    "hipDeviceAttributeCooperativeLaunch",
    "hipDeviceAttributeCooperativeMultiDeviceLaunch",
    "hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim",
    "hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc",
    "hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim",
    "hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem",
    "hipDeviceAttributeCudaCompatibleBegin",
    "hipDeviceAttributeCudaCompatibleEnd",
    "hipDeviceAttributeDeviceOverlap",
    "hipDeviceAttributeDirectManagedMemAccessFromHost",
    "hipDeviceAttributeEccEnabled",
    "hipDeviceAttributeFineGrainSupport",
    "hipDeviceAttributeGlobalL1CacheSupported",
    "hipDeviceAttributeHdpMemFlushCntl",
    "hipDeviceAttributeHdpRegFlushCntl",
    "hipDeviceAttributeHostNativeAtomicSupported",
    "hipDeviceAttributeHostRegisterSupported",
    "hipDeviceAttributeImageSupport",
    "hipDeviceAttributeIntegrated",
    "hipDeviceAttributeIsLargeBar",
    "hipDeviceAttributeIsMultiGpuBoard",
    "hipDeviceAttributeKernelExecTimeout",
    "hipDeviceAttributeL2CacheSize",
    "hipDeviceAttributeLocalL1CacheSupported",
    "hipDeviceAttributeLuid",
    "hipDeviceAttributeLuidDeviceNodeMask",
    "hipDeviceAttributeManagedMemory",
    "hipDeviceAttributeMaxBlockDimX",
    "hipDeviceAttributeMaxBlockDimY",
    "hipDeviceAttributeMaxBlockDimZ",
    "hipDeviceAttributeMaxBlocksPerMultiProcessor",
    "hipDeviceAttributeMaxGridDimX",
    "hipDeviceAttributeMaxGridDimY",
    "hipDeviceAttributeMaxGridDimZ",
    "hipDeviceAttributeMaxPitch",
    "hipDeviceAttributeMaxRegistersPerBlock",
    "hipDeviceAttributeMaxRegistersPerMultiprocessor",
    "hipDeviceAttributeMaxSharedMemoryPerBlock",
    "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
    "hipDeviceAttributeMaxSurface1D",
    "hipDeviceAttributeMaxSurface1DLayered",
    "hipDeviceAttributeMaxSurface2D",
    "hipDeviceAttributeMaxSurface2DLayered",
    "hipDeviceAttributeMaxSurface3D",
    "hipDeviceAttributeMaxSurfaceCubemap",
    "hipDeviceAttributeMaxSurfaceCubemapLayered",
    "hipDeviceAttributeMaxTexture1DLayered",
    "hipDeviceAttributeMaxTexture1DLinear",
    "hipDeviceAttributeMaxTexture1DMipmap",
    "hipDeviceAttributeMaxTexture1DWidth",
    "hipDeviceAttributeMaxTexture2DGather",
    "hipDeviceAttributeMaxTexture2DHeight",
    "hipDeviceAttributeMaxTexture2DLayered",
    "hipDeviceAttributeMaxTexture2DLinear",
    "hipDeviceAttributeMaxTexture2DMipmap",
    "hipDeviceAttributeMaxTexture2DWidth",
    "hipDeviceAttributeMaxTexture3DAlt",
    "hipDeviceAttributeMaxTexture3DDepth",
    "hipDeviceAttributeMaxTexture3DHeight",
    "hipDeviceAttributeMaxTexture3DWidth",
    "hipDeviceAttributeMaxTextureCubemap",
    "hipDeviceAttributeMaxTextureCubemapLayered",
    "hipDeviceAttributeMaxThreadsDim",
    "hipDeviceAttributeMaxThreadsPerBlock",
    "hipDeviceAttributeMaxThreadsPerMultiProcessor",
    "hipDeviceAttributeMemoryBusWidth",
    "hipDeviceAttributeMemoryClockRate",
    "hipDeviceAttributeMemoryPoolsSupported",
    "hipDeviceAttributeMultiGpuBoardGroupID",
    "hipDeviceAttributeMultiprocessorCount",
    "hipDeviceAttributePageableMemoryAccess",
    "hipDeviceAttributePageableMemoryAccessUsesHostPageTables",
    "hipDeviceAttributePciBusId",
    "hipDeviceAttributePciDeviceId",
    "hipDeviceAttributePciDomainID",
    "hipDeviceAttributePersistingL2CacheMaxSize",
    "hipDeviceAttributePhysicalMultiProcessorCount",
    "hipDeviceAttributeReservedSharedMemPerBlock",
    "hipDeviceAttributeSharedMemPerBlockOptin",
    "hipDeviceAttributeSharedMemPerMultiprocessor",
    "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
    "hipDeviceAttributeStreamPrioritiesSupported",
    "hipDeviceAttributeSurfaceAlignment",
    "hipDeviceAttributeTccDriver",
    "hipDeviceAttributeTextureAlignment",
    "hipDeviceAttributeTexturePitchAlignment",
    "hipDeviceAttributeTotalConstantMemory",
    "hipDeviceAttributeTotalGlobalMem",
    "hipDeviceAttributeUnifiedAddressing",
    "hipDeviceAttributeUnused1",
    "hipDeviceAttributeUnused2",
    "hipDeviceAttributeUnused3",
    "hipDeviceAttributeUnused4",
    "hipDeviceAttributeUnused5",
    "hipDeviceAttributeVendorSpecificBegin",
    "hipDeviceAttributeVirtualMemoryManagementSupported",
    "hipDeviceAttributeWallClockRate",
    "hipDeviceAttributeWarpSize",
    "hipDeviceAttribute_t",
    "hipDeviceProp_tR0600",
    "hipDeviceptr_t",
    "hipErrorAlreadyAcquired",
    "hipErrorAlreadyMapped",
    "hipErrorArrayIsMapped",
    "hipErrorAssert",
    "hipErrorCapturedEvent",
    "hipErrorContextAlreadyCurrent",
    "hipErrorContextAlreadyInUse",
    "hipErrorContextIsDestroyed",
    "hipErrorCooperativeLaunchTooLarge",
    "hipErrorDeinitialized",
    "hipErrorECCNotCorrectable",
    "hipErrorFileNotFound",
    "hipErrorGraphExecUpdateFailure",
    "hipErrorHostMemoryAlreadyRegistered",
    "hipErrorHostMemoryNotRegistered",
    "hipErrorIllegalAddress",
    "hipErrorIllegalState",
    "hipErrorInitializationError",
    "hipErrorInsufficientDriver",
    "hipErrorInvalidConfiguration",
    "hipErrorInvalidContext",
    "hipErrorInvalidDevice",
    "hipErrorInvalidDeviceFunction",
    "hipErrorInvalidDevicePointer",
    "hipErrorInvalidGraphicsContext",
    "hipErrorInvalidHandle",
    "hipErrorInvalidImage",
    "hipErrorInvalidKernelFile",
    "hipErrorInvalidMemcpyDirection",
    "hipErrorInvalidPitchValue",
    "hipErrorInvalidResourceHandle",
    "hipErrorInvalidSource",
    "hipErrorInvalidSymbol",
    "hipErrorInvalidValue",
    "hipErrorLaunchFailure",
    "hipErrorLaunchOutOfResources",
    "hipErrorLaunchTimeOut",
    "hipErrorMapBufferObjectFailed",
    "hipErrorMapFailed",
    "hipErrorMemoryAllocation",
    "hipErrorMissingConfiguration",
    "hipErrorNoBinaryForGpu",
    "hipErrorNoDevice",
    "hipErrorNotFound",
    "hipErrorNotInitialized",
    "hipErrorNotMapped",
    "hipErrorNotMappedAsArray",
    "hipErrorNotMappedAsPointer",
    "hipErrorNotReady",
    "hipErrorNotSupported",
    "hipErrorOperatingSystem",
    "hipErrorOutOfMemory",
    "hipErrorPeerAccessAlreadyEnabled",
    "hipErrorPeerAccessNotEnabled",
    "hipErrorPeerAccessUnsupported",
    "hipErrorPriorLaunchFailure",
    "hipErrorProfilerAlreadyStarted",
    "hipErrorProfilerAlreadyStopped",
    "hipErrorProfilerDisabled",
    "hipErrorProfilerNotInitialized",
    "hipErrorRuntimeMemory",
    "hipErrorRuntimeOther",
    "hipErrorSetOnActiveProcess",
    "hipErrorSharedObjectInitFailed",
    "hipErrorSharedObjectSymbolNotFound",
    "hipErrorStreamCaptureImplicit",
    "hipErrorStreamCaptureInvalidated",
    "hipErrorStreamCaptureIsolation",
    "hipErrorStreamCaptureMerge",
    "hipErrorStreamCaptureUnjoined",
    "hipErrorStreamCaptureUnmatched",
    "hipErrorStreamCaptureUnsupported",
    "hipErrorStreamCaptureWrongThread",
    "hipErrorTbd",
    "hipErrorUnknown",
    "hipErrorUnmapFailed",
    "hipErrorUnsupportedLimit",
    "hipError_t",
    "hipFuncGetAttribute",
    "hipFunction_attribute",
    "hipFunction_t",
    "hipGetErrorString",
    "hipJitOption",
    "hipJitOptionCacheMode",
    "hipJitOptionErrorLogBuffer",
    "hipJitOptionErrorLogBufferSizeBytes",
    "hipJitOptionFallbackStrategy",
    "hipJitOptionFastCompile",
    "hipJitOptionGenerateDebugInfo",
    "hipJitOptionGenerateLineInfo",
    "hipJitOptionInfoLogBuffer",
    "hipJitOptionInfoLogBufferSizeBytes",
    "hipJitOptionLogVerbose",
    "hipJitOptionMaxRegisters",
    "hipJitOptionNumOptions",
    "hipJitOptionOptimizationLevel",
    "hipJitOptionSm3xOpt",
    "hipJitOptionTarget",
    "hipJitOptionTargetFromContext",
    "hipJitOptionThreadsPerBlock",
    "hipJitOptionWallTime",
    "hipMemoryType",
    "hipMemoryTypeArray",
    "hipMemoryTypeDevice",
    "hipMemoryTypeHost",
    "hipMemoryTypeManaged",
    "hipMemoryTypeUnified",
    "hipMemoryTypeUnregistered",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
    "hipModuleLoadDataEx",
    "hipModule_t",
    "hipPointerAttribute_t",
    "hipPointerGetAttributes",
    "hipStream_t",
    "hipSuccess",
    "hipUUID",
]
