#include <algorithm>
#include <alloca.h>
#include <dlfcn.h>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>

#include "cuda.h"

namespace py = pybind11;

namespace {

// Raise a python exception if the CUDA result code is not CUDA_SUCCESS.
inline void gpuAssert(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return;
  const char *errorPtr = nullptr;
  cuGetErrorString(code, &errorPtr);
  std::string errorStr = errorPtr ? errorPtr : "Unknown CUDA error";
  std::string errorMessage = "Triton Error [CUDA]: " + errorStr;
  throw py::cast_error(errorMessage);
}

#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__);

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == nullptr) {                                            \
      (funcPointer) = (initializerFunction)();                                 \
    }                                                                          \
  } while (0)

#define GET_FUNCTION_HANDLE(name, symbolName)                                  \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    /* Check for errors */                                                     \
    if (dlerror()) {                                                           \
      throw py::value_error("Failed to open libcuda.so.1");                    \
      return nullptr;                                                          \
    }                                                                          \
    symbolName##_t funcHandle =                                                \
        reinterpret_cast<symbolName##_t>(dlsym(libHandle, #symbolName));       \
    /* Check for errors */                                                     \
    if (dlerror()) {                                                           \
      dlclose(libHandle);                                                      \
      throw py::value_error("Failed to retrieve " #symbolName                  \
                            " from libcuda.so.1");                             \
      return nullptr;                                                          \
    }                                                                          \
    return funcHandle;                                                         \
  }

typedef CUresult (*cuOccupancyMaxActiveClusters_t)(
    int *numClusters, CUfunction func, const CUlaunchConfig *config);
GET_FUNCTION_HANDLE(getCuOccupancyMaxActiveClustersHandle,
                    cuOccupancyMaxActiveClusters);

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);
GET_FUNCTION_HANDLE(getCuTensorMapEncodeTiledHandle, cuTensorMapEncodeTiled);

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig *config,
                                       CUfunction f, void **kernelParams,
                                       void **extra);
GET_FUNCTION_HANDLE(getLaunchKernelExHandle, cuLaunchKernelEx);

// Configuration with all the information necessary to launch a compiled
// Triton kernel using the CUDA driver API.
struct TritonLaunchConfig {
  // Represents CUDA's 3D ID structure of grids and clusters
  struct Dim {
    unsigned x = 1;
    unsigned y = 1;
    unsigned z = 1;
    constexpr unsigned size() const { return x * y * z; }
  };
  Dim grid;                      // Number of clusters per grid
  Dim cluster;                   // Number of blocks per cluster
  int numWarps = 0;              // Number of warps per block
  int sharedMemory = 0;          // Size of shared memory in bytes to allocate
  int launchCooperativeGrid = 0; // Non-zero to launch coop grid
  int launchPdl = 0;             // Non-zero for programmatic-dependent launch
  CUstream stream = nullptr;     // CUDA Stream on which to launch the kernel
  CUfunction function = nullptr; // Pointer to the kernel to launch
  void **params = nullptr;       // Parameters to pass to the kernel
};

// Launch a CUDA kernel with the given parameters. Raises a Python exception
// if the kernel launch fails.
void launchKernel(const TritonLaunchConfig &config) {
  // Launching the kernel might take a while and does not use Python APIs, so
  // we can release the Global Interpreter Lock so other threads can use Python
  // APIs if needed.
  py::gil_scoped_release release;
  const auto &grid = config.grid;
  const auto &cluster = config.cluster;
  if (grid.size() == 0) {
    return;
  }
  CUlaunchConfig cuConfig;
  cuConfig.gridDimX = grid.x * cluster.x;
  cuConfig.gridDimY = grid.y * cluster.y;
  cuConfig.gridDimZ = grid.z * cluster.z;
  cuConfig.blockDimX = 32 * config.numWarps;
  cuConfig.blockDimY = 1;
  cuConfig.blockDimZ = 1;
  cuConfig.sharedMemBytes = config.sharedMemory;
  cuConfig.hStream = config.stream;

  // We support passing up to 4 attributes.
  CUlaunchAttribute launchAttr[4];
  cuConfig.attrs = launchAttr;
  cuConfig.numAttrs = 0;
  if (config.launchPdl != 0) {
    launchAttr[cuConfig.numAttrs++] = CUlaunchAttribute{
        .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        .value = {.programmaticStreamSerializationAllowed = 1}};
  }
  if (config.launchCooperativeGrid != 0) {
    launchAttr[cuConfig.numAttrs++] = CUlaunchAttribute{
        .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = {.cooperative = 1}};
  }
  if (config.cluster.size() > 1) {
    launchAttr[cuConfig.numAttrs++] = CUlaunchAttribute{
        .id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
        .value = {
            .clusterDim = {.x = cluster.x, .y = cluster.y, .z = cluster.z}}};
    launchAttr[cuConfig.numAttrs++] = CUlaunchAttribute{
        .id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
        .value = {.clusterSchedulingPolicyPreference =
                      CU_CLUSTER_SCHEDULING_POLICY_SPREAD}};
  }
  // cuLaunchKernelEx was added in CUDA 12, so load it dynamically to be
  // able to link on CUDA 11 and earlier.
  static cuLaunchKernelEx_t cuLaunchKernelEx = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuLaunchKernelEx,
                                      getLaunchKernelExHandle);

  CUDA_CHECK(cuLaunchKernelEx(&cuConfig, config.function, config.params, 0));
}

// Interface used by various py::object extractors to extract obj into a memory
// location pointed by ptr. Returns true if extraction succeeded, and false
// otherwise.
using ExtractorType = void (*)(py::handle obj, void *ptr);

// Extract a CUDA device pointer from a pointer-like py::object obj, and store
// it to the memory location pointed by ptr.
void extractPointer(py::handle obj, void *ptr) {
  auto devPtr = static_cast<CUdeviceptr *>(ptr);
  if (obj.is_none()) {
    *devPtr = static_cast<CUdeviceptr>(0); // valid nullptr
    return;
  }
  if (py::isinstance<py::int_>(obj)) {
    *devPtr = obj.cast<uint64_t>();
    return;
  }
  if (!py::hasattr(obj, "data_ptr")) {
    py::str errorMsg = py::str("Pointer argument must be either uint64 or have "
                               "data_ptr method, but got {0}")
                           .format(obj);
    throw py::type_error(errorMsg);
  }

  py::object ret = obj.attr("data_ptr")();
  if (!py::isinstance<py::int_>(ret)) {
    throw py::type_error(
        "data_ptr method of Pointer object must return 64-bit int");
  }
  *devPtr = ret.cast<uint64_t>();
  if (*devPtr == 0) {
    return; // valid nullptr
  }

  CUresult status = cuPointerGetAttribute(
      devPtr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *devPtr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
    throw py::value_error(
        "Pointer argument cannot be accessed from Triton (cpu tensor?)");
  }
  CUDA_CHECK(status);
}

// Extract a CUtensorMap descriptor from a python object, and store it to the
// memory location pointed by ptr.
void extractTmaDesc(py::handle obj, void *ptr) {
  if (sizeof(CUtensorMap *) != 8) {
    throw py::type_error("extractTmaDesc() requires 64-bit compilation");
  }

  if (!py::hasattr(obj, "tma_desc_cpu_ptr")) {
    throw py::type_error(
        "Type nvTmaDesc must have callable method tma_desc_cpu_ptr");
  }
  py::object ret = obj.attr("tma_desc_cpu_ptr")();
  if (!py::isinstance<py::int_>(ret)) {
    throw py::type_error(
        "tma_desc_cpu_ptr method of Pointer object must return 64-bit int");
  }
  uint64_t ptrAsUint = ret.cast<uint64_t>();

  if (!ptrAsUint) {
    throw py::value_error("received NULL ptr from tma_desc_cpu_ptr()");
  }
  if (ptrAsUint % 64 != 0) {
    throw py::value_error("tma_desc_cpu_ptr() must be 64-byte aligned");
  }

  *static_cast<CUtensorMap *>(ptr) =
      *reinterpret_cast<CUtensorMap *>(ptrAsUint);
}

void extractfp16(py::handle obj, void *ptr) {
  double temp_double = obj.cast<double>();
  uint16_t result;
  // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 &&            \
    !defined(PYPY_VERSION)
  _PyFloat_Pack2(temp_double, (unsigned char *)&result, 1);
#else
  PyFloat_Pack2(temp_double, (char *)&result, 1);
#endif
  *static_cast<uint16_t *>(ptr) = result;
}

void extractbf16(py::handle obj, void *ptr) {
  double temp_double = obj.cast<double>();
  float temp_float = (float)temp_double;
  uint32_t u32 = *(uint32_t *)&temp_float;
  *static_cast<uint16_t *>(ptr) = (u32 >> 16);
}

void extractfp32(py::handle obj, void *ptr) {
  double temp_double = obj.cast<double>();
  float temp_float = (float)temp_double;
  uint32_t u32 = *(uint32_t *)&temp_float;
  *static_cast<uint32_t *>(ptr) = u32;
}

void extractfp64(py::handle obj, void *ptr) {
  double temp = obj.cast<double>();
  *static_cast<uint64_t *>(ptr) = *(uint64_t *)&temp;
}

// Extract a value of type T from obj and store it into memory pointed by ptr.
// Returns true if extraction succeeded, and false otherwise.
template <typename T> void extractValue(py::handle obj, void *ptr) {
  *static_cast<T *>(ptr) = obj.cast<T>();
}

// Contains information necessary for extracting a certain type from a
// py::object.
struct ExtractionInfo {
  // Prefixes of types reprs supported by the extractor.
  std::string_view prefix;
  std::size_t size;        // Size required by the extracted value.
  ExtractorType extractor; // Function to call to extract the value.

  // Builds an ExtractionInfo for a given type T and a list of type reprs that
  // are backed by that type.
  template <typename T>
  static ExtractionInfo build(std::string_view prefix,
                              ExtractorType extractor = extractValue<T>) {
    return ExtractionInfo{prefix, sizeof(T), extractor};
  }

  // Checks if the extractor supports extracting a given type repr.
  bool supports(std::string_view typeRepr) const {
    return typeRepr.length() >= prefix.length() &&
           typeRepr.substr(0, prefix.length()) == prefix;
  }
};

// Array of supported extractors
const ExtractionInfo kExtractionInfos[]{
    ExtractionInfo::build<std::int8_t>("i8"),
    ExtractionInfo::build<std::int16_t>("i16"),
    ExtractionInfo::build<std::int32_t>("i32"),
    ExtractionInfo::build<std::int32_t>("i1"),
    ExtractionInfo::build<std::int64_t>("i64"),
    ExtractionInfo::build<std::uint8_t>("u8"),
    ExtractionInfo::build<std::uint16_t>("u16"),
    ExtractionInfo::build<std::uint32_t>("u1"),
    ExtractionInfo::build<std::uint32_t>("u32"),
    ExtractionInfo::build<std::uint64_t>("u64"),
    ExtractionInfo::build<uint16_t>("fp16", extractfp16),
    ExtractionInfo::build<uint16_t>("bf16", extractbf16),
    ExtractionInfo::build<uint32_t>("fp32", extractfp32),
    ExtractionInfo::build<uint32_t>("f32", extractfp32),
    ExtractionInfo::build<uint64_t>("fp64", extractfp64),
    ExtractionInfo::build<void *>("*", extractPointer),
    ExtractionInfo{"None", 0, nullptr}, // Represent constexprs as None
    ExtractionInfo{"none", 0, nullptr}, // Represent constexprs as None
    ExtractionInfo::build<CUtensorMap>({"nvTmaDesc"}, extractTmaDesc),
};

using ExtractorIndex = uint8_t;
// Finds an extractor that supports a given type_repr in the extractor list.
// Returns nullopt if no such extractor is found.
std::optional<ExtractorIndex> findExtractor(std::string_view typeRepr) {
  constexpr std::size_t kNumExtractors = std::size(kExtractionInfos);
  static_assert(kNumExtractors < std::numeric_limits<ExtractorIndex>::max(),
                "Not enough bits in a byte to store the extractor index");
  for (ExtractorIndex i = 0; i < kNumExtractors; ++i) {
    if (kExtractionInfos[i].supports(typeRepr))
      return i;
  }
  return std::nullopt;
}

std::vector<ExtractorIndex>
buildSignatureMetadata(const std::vector<std::string> &signature) {
  std::vector<ExtractorIndex> signatureMetadata;
  signatureMetadata.reserve(signature.size());
  for (std::string type : signature) {
    std::optional<ExtractorIndex> extractorIdx = findExtractor(type);
    if (!extractorIdx.has_value()) {
      throw py::type_error(
          py::str("unexpected type {0} in kernel signature").format(type));
    }
    signatureMetadata.push_back(extractorIdx.value());
  }
  return signatureMetadata;
}

// Launch a Python callable hook with metadata passed as parameters.
void launchHook(py::object hook, py::object metadata) {
  if (hook.is_none()) {
    return;
  }
  py::tuple args = py::make_tuple(metadata);
  py::object ret = hook(*args);
}

void ensureCudaContext(CUdevice *device) {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUDA_CHECK(cuDeviceGet(device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, *device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

void launch(
    unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ, int64_t stream,
    int64_t function, bool launchCooperativeGrid, bool launchPdl,
    /* packedMetadata: 6-tuple of (number of warps, number of CTAs,
     * required bytes of shared memory, cluster dimension x, y, and
     * z) */
    std::tuple<int, int, int, unsigned, unsigned, unsigned> packedMetadata,
    py::object hookArgs, py::object launchEnterHook, py::object launchExitHook,
    py::iterable signatureMetadata, py::object globalScratch,
    py::iterable kernelArgs) {
  CUdevice device;
  ensureCudaContext(&device);
  auto &[numWarps, /*numCtas*/ _, bytesShared, clusterX, clusterY, clusterZ] =
      packedMetadata;
  TritonLaunchConfig config{
      .grid = {gridDimX, gridDimY, gridDimZ},
      .cluster = {clusterX, clusterY, clusterZ},
      .numWarps = numWarps,
      .sharedMemory = bytesShared,
      .launchCooperativeGrid = launchCooperativeGrid,
      .launchPdl = launchPdl,
      .stream = reinterpret_cast<CUstream>(stream),
      .function = reinterpret_cast<CUfunction>(function),
  };

  std::size_t numParams = py::len(signatureMetadata);
  if (numParams != py::len(kernelArgs)) {
    throw py::type_error(
        py::str("Expected kernel to have {0} parameters, but got {1}")
            .format(numParams, py::len(kernelArgs)));
  }
  numParams++; // +1 for the global scratch pointer.

  // Use alloca to set up kernel parameters on the stack and avoid dynamic
  // memory allocations.
  config.params = static_cast<void **>(alloca(numParams * sizeof(void *)));
  // This loop has to stay in the same function that owns params, since we are
  // using alloca to allocate pointers to it on the stack of the function.
  std::size_t paramsIdx = 0;
  py::iterator arg_data = py::iter(signatureMetadata);
  for (py::handle arg : kernelArgs) {
    ExtractorIndex converterIdx = arg_data->cast<ExtractorIndex>();
    arg_data++;
    if (converterIdx >= std::size(kExtractionInfos)) {
      throw py::value_error("corrupted signature metadata");
    }
    const ExtractionInfo &extraction_info = kExtractionInfos[converterIdx];
    if (extraction_info.size == 0) {
      continue; // skip adding constexpr parameters
    }
    config.params[paramsIdx] = alloca(extraction_info.size);
    extraction_info.extractor(arg, config.params[paramsIdx]);
    ++paramsIdx;
  }
  config.params[paramsIdx] = alloca(sizeof(void *));
  extractPointer(globalScratch, config.params[paramsIdx]);

  launchHook(launchEnterHook, hookArgs);

  launchKernel(config);

  launchHook(launchExitHook, hookArgs);
}

std::map<std::string, int32_t> getDeviceProperties(int32_t deviceId) {
  // Get device handle
  CUdevice device;
  cuDeviceGet(&device, deviceId);

  int32_t maxSharedMem;
  CUDA_CHECK(cuDeviceGetAttribute(
      &maxSharedMem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  int32_t maxNumRegs;
  CUDA_CHECK(cuDeviceGetAttribute(
      &maxNumRegs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  int32_t multiprocessorCount;
  CUDA_CHECK(cuDeviceGetAttribute(
      &multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  int32_t warpSize;
  CUDA_CHECK(
      cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  int32_t smClockRate;
  CUDA_CHECK(cuDeviceGetAttribute(&smClockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                  device));
  int32_t memClockRate;
  CUDA_CHECK(cuDeviceGetAttribute(
      &memClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  int32_t memBusWidth;
  CUDA_CHECK(cuDeviceGetAttribute(
      &memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  std::map<std::string, int32_t> properties = {
      {"max_shared_mem", maxSharedMem},
      {"max_num_regs", maxNumRegs},
      {"multiprocessor_count", multiprocessorCount},
      {"warpSize", warpSize},
      {"sm_clock_rate", smClockRate},
      {"mem_clock_rate", memClockRate},
      {"mem_bus_width", memBusWidth},
  };
  return properties;
}

std::tuple<uint64_t, uint64_t, int32_t, int32_t, int32_t>
loadBinary(char *name, char *data, int shared, CUdevice device) {
  py::gil_scoped_release release;
  ensureCudaContext(&device);

  CUmodule mod;
  CUDA_CHECK(cuModuleLoadData(&mod, data));
  CUfunction fun;
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));
  // Get allocated registers and spilled registers from the function.
  int32_t nRegs = 0;
  CUDA_CHECK(cuFuncGetAttribute(&nRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  int32_t nSpills = 0;
  CUDA_CHECK(
      cuFuncGetAttribute(&nSpills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  nSpills /= 4;
  int32_t nMaxThreads = 0;
  CUDA_CHECK(cuFuncGetAttribute(&nMaxThreads,
                                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));

  int32_t sharedOptin;
  CUDA_CHECK(cuDeviceGetAttribute(
      &sharedOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  // If the amount of shared memory is higher than the static shared memory
  // limit, then set dynamic shared memory to the difference.
  constexpr int32_t kStaticSharedMemLim = 49152;
  if (shared > kStaticSharedMemLim && sharedOptin > kStaticSharedMemLim) {
    CUDA_CHECK(cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
    int32_t sharedStatic;
    CUDA_CHECK(cuFuncGetAttribute(&sharedStatic,
                                  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    CUDA_CHECK(
        cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           sharedOptin - sharedStatic));
  }

  return {reinterpret_cast<uint64_t>(mod), reinterpret_cast<uint64_t>(fun),
          nRegs, nSpills, nMaxThreads};
}

int32_t occupancyMaxActiveClusters(uint64_t func, uint32_t shared,
                                   uint32_t clusterDimX, uint32_t clusterDimY,
                                   uint32_t clusterDimZ) {
  CUfunction cuFunc = (CUfunction)func;

  // Let each SM have one block
  py::gil_scoped_release release;
  CUDA_CHECK(cuFuncSetAttribute(
      cuFunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));

  CUlaunchAttribute launchAttr{
      .id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
      .value = {
          .clusterDim{.x = clusterDimX, .y = clusterDimY, .z = clusterDimZ}}};

  constexpr int32_t maxActiveBlocks = 1;
  CUlaunchConfig config;
  config.gridDimX = clusterDimX;
  config.gridDimY = maxActiveBlocks * clusterDimY;
  config.gridDimZ = clusterDimZ;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared;
  config.hStream = 0;
  config.numAttrs = 1;
  config.attrs = &launchAttr;

  static cuOccupancyMaxActiveClusters_t cuOccupancyMaxActiveClusters = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuOccupancyMaxActiveClusters,
                                      getCuOccupancyMaxActiveClustersHandle);

  CUDA_CHECK(cuFuncSetAttribute(
      cuFunc, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  int32_t maxActiveClusters = -1;
  CUDA_CHECK(cuOccupancyMaxActiveClusters(&maxActiveClusters, cuFunc, &config));
  return maxActiveClusters;
}

void setPrintfFifoSize(int32_t size) {
  if (size < 0) {
    throw py::value_error("fifo size must be non-negative");
  }

  py::gil_scoped_release release;

  // Ensure we have an active context.
  CUdevice device;
  ensureCudaContext(&device);

  // We can't set the fifo size after running a kernel that calls printf.  This
  // is true even if the set() call is a nop and the new size is the same as the
  // old size.
  //
  // This is unfriendly, so check if the old size matches the new size, and skip
  // the set() call if so.
  size_t oldSize = 0;
  CUDA_CHECK(cuCtxGetLimit(&oldSize, CU_LIMIT_PRINTF_FIFO_SIZE));
  if (oldSize != size) {
    CUDA_CHECK(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, size));
  }
}

void fillTMADescriptor(uint64_t descAddress, uint64_t globalAddress,
                       int swizzle, int elemSize, int elemType,
                       const std::vector<int64_t> &blockSize,
                       const std::vector<int32_t> &shape,
                       const std::vector<int64_t> &strides) {
  constexpr int kMaxRank = 5;
  uint32_t blockSizeInt[kMaxRank];

  int rank = blockSize.size();
  if (rank > kMaxRank) {
    throw py::value_error(
        py::str("Rank higher than max rank: {0} > {1}").format(rank, kMaxRank));
  }

  for (int i = 0; i < rank; ++i) {
    blockSizeInt[rank - i - 1] = blockSize[i];
  }

  if (rank != shape.size()) {
    throw py::value_error("Rank mismatch");
  }
  uint64_t shapeInt[kMaxRank];
  for (int i = 0; i < rank; ++i) {
    shapeInt[rank - i - 1] = shape[i];
  }

  if (rank != strides.size()) {
    throw py::value_error("Rank mismatch");
  }
  uint64_t stridesLL[kMaxRank];
  for (int i = 0; i + 1 < rank; ++i) {
    stridesLL[rank - i - 2] = elemSize * strides[i];
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);

  uint32_t elementStrides[kMaxRank] = {1, 1, 1, 1, 1};
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK(cuTensorMapEncodeTiled(
      (CUtensorMap *)descAddress, (CUtensorMapDataType)elemType, rank,
      (void *)globalAddress, shapeInt, stridesLL, blockSizeInt, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, (CUtensorMapSwizzle)swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  return;
}

} // namespace

void init_triton_cuda_utils(py::module &&m) {
  m.def("load_binary", &loadBinary);
  m.def("get_device_properties", &getDeviceProperties);
  m.def("cuOccupancyMaxActiveClusters", &occupancyMaxActiveClusters);
  m.def("set_printf_fifo_size", &setPrintfFifoSize);
  m.def("fill_tma_descriptor", &fillTMADescriptor);
  m.def("build_signature_metadata", &buildSignatureMetadata);
  m.def("launch", &launch);
}
