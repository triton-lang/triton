#include <algorithm>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <alloca.h>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>

#include "cuda.h"

namespace py = pybind11;

namespace {

// Raise a python exception if the CUDA result code is not CUDA_SUCCESS.
inline bool gpuAssert(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return true;
  const char *error = nullptr;
  cuGetErrorString(code, &error);
  py::str error_str = py::str("Triton Error [CUDA]: {0}").format(error);
  throw pybind11::cast_error(error_str);
}

#define CUDA_CHECK(ans)                                                        \
  {{gpuAssert((ans), __FILE__, __LINE__);                                      \
  }                                                                            \
  }

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        throw py::value_error("Failed to initialize function pointer");        \
      }                                                                        \
    }                                                                          \
  } while (0)

using cuLaunchKernelEx_t = CUresult (*)(const CUlaunchConfig *config,
                                        CUfunction f, void **kernelParams,
                                        void **extra);

// Dynamically load the handle to cuLaunchKernelEx.
cuLaunchKernelEx_t getLaunchKernelExHandle() {
  // Open the shared library
  void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!handle) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");
    return nullptr;
  }
  // Clear any existing error
  dlerror();
  auto cuLaunchKernelExHandle =
      reinterpret_cast<cuLaunchKernelEx_t>(dlsym(handle, "cuLaunchKernelEx"));
  // Check for errors
  if (const char *dlsym_error = dlerror()) {
    PyErr_Format(PyExc_RuntimeError,
                 "Failed to retrieve cuLaunchKernelEx from libcuda.so: %s",
                 dlsym_error);
    return nullptr;
  }
  return cuLaunchKernelExHandle;
}

// Configuration with all the information necessary to launch a compiled
// Triton kernel using the CUDA driver API.
struct TritonLaunchConfig {
  // Represents CUDA's 3D ID structure of grids and clusters
  struct Dim {
    int x;
    int y;
    int z;
    constexpr int size() const { return x * y * z; }
  };
  Dim grid;                    // Number of clusters per grid
  Dim cluster;                 // Number of blocks per cluster
  int num_warps;               // number of warps per block
  int shared_memory;           // Size of shared memory in bytes to allocate
  int launch_cooperative_grid; // Non-zero to launch coop grid
  int launch_pdl;              // Non-zero to use programmatic-dependent launch
  CUstream stream;             // CUDA Stream on which to launch the kernel
  CUfunction function;         // Pointer to the kernel to launch
  void **params;               // Parameters to pass to the kernel
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
  CUlaunchConfig cu_config;
  cu_config.gridDimX = grid.x * cluster.x;
  cu_config.gridDimY = grid.y * cluster.y;
  cu_config.gridDimZ = grid.z * cluster.z;
  cu_config.blockDimX = 32 * config.num_warps;
  cu_config.blockDimY = 1;
  cu_config.blockDimZ = 1;
  cu_config.sharedMemBytes = config.shared_memory;
  cu_config.hStream = config.stream;

  // We support passing up to 4 attributes.
  CUlaunchAttribute launchAttr[4];
  cu_config.attrs = launchAttr;
  cu_config.numAttrs = 0;
  if (config.launch_pdl != 0) {
    launchAttr[cu_config.numAttrs++] = CUlaunchAttribute{
        .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        .value = {.programmaticStreamSerializationAllowed = 1}};
  }
  if (config.launch_cooperative_grid != 0) {
    launchAttr[cu_config.numAttrs++] = CUlaunchAttribute{
        .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = {.cooperative = 1}};
  }
  if (config.cluster.size() > 1) {
    auto &clusterDimAttr = launchAttr[cu_config.numAttrs++];
    clusterDimAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    clusterDimAttr.value.clusterDim.x = cluster.x;
    clusterDimAttr.value.clusterDim.y = cluster.y;
    clusterDimAttr.value.clusterDim.z = cluster.z;
    auto &clusterDimSchedulingAttr = launchAttr[cu_config.numAttrs++];
    clusterDimSchedulingAttr.id =
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    clusterDimSchedulingAttr.value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
  }
  // cuLaunchKernelEx was added in CUDA 12, so load it dynamically to be
  // able to link on CUDA 11 and earlier.
  static cuLaunchKernelEx_t cuLaunchKernelExHandle = getLaunchKernelExHandle();
  CUDA_CHECK(
      cuLaunchKernelExHandle(&cu_config, config.function, config.params, 0));
}

// Interface used by various py::object extractors to extract obj into a memory
// location pointed by ptr. Returns true if extraction succeeded, and false
// otherwise.
using ExtractorType = bool (*)(py::object obj, void *ptr);

// Enable peer access if dev_ptr is allocated on a different device than the
// device on which we will execute the kernel.
void enablePeerAccessIfNecessary(CUdeviceptr dev_ptr) {
  CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
  CUresult status = cuPointerGetAttribute(
      &mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dev_ptr);
  if (status != CUDA_SUCCESS || mem_type != CU_MEMORYTYPE_DEVICE) {
    return;
  }
  int mem_device_ordinal = 0;
  CUDA_CHECK(cuPointerGetAttribute(
      &mem_device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dev_ptr));
  CUdevice mem_device = 0;
  CUDA_CHECK(cuDeviceGet(&mem_device, mem_device_ordinal));
  CUdevice compute_device = 0;
  CUDA_CHECK(cuCtxGetDevice(&compute_device));
  if (mem_device != compute_device) {
    CUcontext mem_ctx = nullptr;
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&mem_ctx, mem_device));
    CUresult status = cuCtxEnablePeerAccess(mem_ctx, /*flags=*/0);
    if (status == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
      status = CUDA_SUCCESS;
    }
    CUDA_CHECK(status);
  }
}

// Extract a CUDA device pointer from a pointer-like py::object obj, and store
// it to the memory location pointed by ptr.
bool extractPointer(py::object obj, void *ptr) {
  auto dev_ptr = static_cast<CUdeviceptr *>(ptr);
  if (obj.is_none()) {
    *dev_ptr = static_cast<CUdeviceptr>(0); // valid nullptr
    return true;
  }
  if (py::isinstance<py::int_>(obj)) {
    *dev_ptr = obj.cast<uint64_t>();
    return true;
  }
  if (!py::hasattr(obj, "data_ptr")) {
    py::str error_msg =
        py::str("Pointer argument must be either uint64 or have "
                "data_ptr method, but got {0}")
            .format(obj);
    throw py::type_error(error_msg);
  }

  py::object ret = obj.attr("data_ptr")();
  if (!py::isinstance<py::int_>(ret)) {
    throw py::type_error(
        "data_ptr method of Pointer object must return 64-bit int");
  }
  *dev_ptr = ret.cast<uint64_t>();
  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  enablePeerAccessIfNecessary(*dev_ptr);

  CUresult status = cuPointerGetAttribute(
      dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
    throw py::value_error(
        "Pointer argument cannot be accessed from Triton (cpu tensor?)");
  } else if (status != CUDA_SUCCESS) {
    CUDA_CHECK(status);
    return false;
  }
  return true;
}

// Extract a CUtensorMap descriptor from a python object, and store it to the
// memory location pointed by ptr.
bool extractTmaDesc(py::object obj, void *ptr) {
  if (sizeof(CUtensorMap *) != 8) {
    PyErr_SetString(PyExc_SystemError,
                    "extractTmaDesc() requires 64-bit compilation");
    return false;
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
  uint64_t ptr_as_uint = ret.cast<uint64_t>();

  if (!ptr_as_uint) {
    throw py::value_error("received NULL ptr from tma_desc_cpu_ptr()");
  }
  if (ptr_as_uint % 64 != 0) {
    throw py::value_error("tma_desc_cpu_ptr() must be 64-byte aligned");
  }

  *static_cast<CUtensorMap *>(ptr) =
      *reinterpret_cast<CUtensorMap *>(ptr_as_uint);
  return true;
}

// Extract a value of type T from obj and store it into memory pointed by ptr.
// Returns true if extraction succeeded, and false otherwise.
template <typename T> bool extractValue(py::object obj, void *ptr) {
  *static_cast<T *>(ptr) = obj.cast<T>();
  return true;
}

// Contains information necessary for extracting a certain type from a
// py::object.
struct ExtractionInfo {
  // Prefixes of types reprs supported by the extractor.
  std::vector<std::string_view> supported_type_repr_prefixes;
  std::size_t size;        // Size required by the extracted value.
  ExtractorType extractor; // Function to call to extract the value.

  // Builds an ExtractionInfo for a given type T and a list of type reprs that
  // are backed by that type.
  template <typename T>
  static ExtractionInfo
  build(std::initializer_list<std::string_view> supported_type_reprs,
        ExtractorType extractor = extractValue<T>) {
    return {supported_type_reprs, sizeof(T), extractor};
  }

  // Checks if the extractor supports extracting a given type repr.
  bool supports(std::string_view type_repr) const {
    return std::any_of(supported_type_repr_prefixes.begin(),
                       supported_type_repr_prefixes.end(),
                       [&](std::string_view prefix) {
                         return type_repr.length() >= prefix.length() &&
                                type_repr.substr(0, prefix.length()) == prefix;
                       });
  }
};

// Array of supported extractors
const ExtractionInfo kExtractionInfos[]{
    ExtractionInfo::build<std::int8_t>({"i8"}),
    ExtractionInfo::build<std::int16_t>({"i16"}),
    ExtractionInfo::build<std::int32_t>({"i1", "i32"}),
    ExtractionInfo::build<std::int64_t>({"i64"}),
    ExtractionInfo::build<std::uint8_t>({"u8"}),
    ExtractionInfo::build<std::uint16_t>({"u16"}),
    ExtractionInfo::build<std::uint32_t>({"u1", "u32"}),
    ExtractionInfo::build<std::uint64_t>({"u64"}),
    ExtractionInfo::build<float>({"fp16", "bf16", "fp32", "f32"}),
    ExtractionInfo::build<double>({"fp64"}),
    ExtractionInfo::build<void *>({"*"}, extractPointer),
    ExtractionInfo{
        {"None", "none"}, 0, nullptr}, // Represent constexprs as None
    ExtractionInfo::build<CUtensorMap>({"nvTmaDesc"}, extractTmaDesc),
};

// Finds an extractor that supports a given type_repr in the extractor list.
// Returns nullopt if no such extractor is found.
std::optional<char> findExtractor(std::string_view type_repr) {
  constexpr std::size_t kNumExtractors = std::size(kExtractionInfos);
  static_assert(kNumExtractors < std::numeric_limits<char>::max(),
                "Not enough bits in a byte to store the extractor index");
  for (int i = 0; i < kNumExtractors; ++i) {
    if (kExtractionInfos[i].supports(type_repr))
      return i;
  }
  return std::nullopt;
}

PyDoc_STRVAR(buildSignatureMetadata__doc__,
             R"(buildSignatureMetadata(signature_iterator) -> bytes

Build a metadata object describing the signature of a kernel.

This can then be passed as the signature_metadata parameter to the launch()
function.

:param signature: list of types describing the signature of a kernel,
    specialized parameters should be represented with None
:type signature: sequence or iterable
:return: an opaque metadata object which can then be passed to launch()
:rtype: bytes
)");
std::vector<char> buildSignatureMetadata(std::vector<std::string> signature) {
  std::vector<char> signature_metadata;
  for (std::string type : signature) {
    std::optional<std::uint8_t> extractor_idx = findExtractor(type);
    if (!extractor_idx.has_value()) {
      throw pybind11::type_error(
          py::str("unexpected type {0} in kernel signature").format(type));
    }
    signature_metadata.push_back(extractor_idx.value());
  }
  return signature_metadata;
}

// Launch a Python callable hook with metadata passed as parameters.
bool launchHook(py::object hook, py::object metadata) {
  if (hook.is_none()) {
    return true;
  }
  py::tuple args = py::make_tuple(metadata);
  py::object ret = hook(*args);
  return ret.cast<bool>();
}

static void ensureCudaContext() {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

PyDoc_STRVAR(
    launch__doc__,
    R"(launch(gridDimX, gridDimY, gridDimZ, stream, kernel, packed_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, kernel_arg_types, global_scratch, kernel_args)

Launch a kernel on an Nvidia GPU.

:param gridDimX: X dimension of the grid
:type gridDimX: signed integer
:param gridDimY: Y dimension of the grid
:type gridDimY: signed integer
:param gridDimZ: Z dimension of the grid
:type gridDimZ: signed integer
:param stream: CUDA Stream to launch on
:type stream: unsigned long integer (pointer)
:param kernel: CUDA kernel to launch
:type kernel: unsigned long integer (pointer)
:param launch_cooperative_grid: launch a cooperative grid
:type launch_cooperative_grid bool
:param launch_pdl: enable programmatic dependent launch
:type launch_pdl: bool
:param packed_metadata: Kernel metadata, including in sequence:
    number of warps, number of CTAs, required bytes of shared memory,
    cluster dimensions x, y, and z
:type packed_metadata: 6-tuple
:param hook_args: arguments to pass to the enter and exit hooks
:type hook_args: object
:param launch_enter_hook: hook to call just before launching the kernel
:type launch_enter_hook: callable
:param launch_exit_hook: hook to call just after launching the kernel
:type launch_exit_hook: callable
:param signature_metadata: matadata built from build_signature_metadata
:type signature_metadata: bytes
:param global_scratch: pointer to global scratch memory
:type global_scratch: pointer
:param kernel_args: kernel parameters
:type kernel_args: tuple

:raises RuntimeError: on kernel launch failure
)");
void launch(int grid_dim_x, int grid_dim_y, int grid_dim_z, int64_t stream,
            int64_t function, bool launch_cooperative_grid, bool launch_pdl,
            std::tuple<int, int, int, int, int, int> packed_metadata,
            py::object hook_args, py::object launch_enter_hook,
            py::object launch_exit_hook, std::vector<char> signature_metadata,
            py::object global_scratch, std::vector<py::object> kernel_args) {
  ensureCudaContext();
  TritonLaunchConfig config{
      .grid = {grid_dim_x, grid_dim_y, grid_dim_z},
      .cluster = {std::get<3>(packed_metadata), std::get<4>(packed_metadata),
                  std::get<5>(packed_metadata)},
      .num_warps = std::get<0>(packed_metadata),
      .shared_memory = std::get<2>(packed_metadata),
      .launch_cooperative_grid = launch_cooperative_grid,
      .launch_pdl = launch_pdl,
      .stream = reinterpret_cast<CUstream>(stream),
      .function = reinterpret_cast<CUfunction>(function),
  };

  if (signature_metadata.size() != kernel_args.size()) {
    throw py::type_error(
        py::str("Expected kernel to have {0} parameters, but got {1}")
            .format(signature_metadata.size(), kernel_args.size()));
  }

  // +1 for the global scratch pointer.
  std::size_t num_params = signature_metadata.size() + 1;
  // Use alloca to set up kernel parameters on the stack and avoid dynamic
  // memory allocations.
  config.params = static_cast<void **>(alloca(num_params * sizeof(void *)));
  // This loop has to stay in the same function that owns params, since we are
  // using alloca to allocate pointers to it on the stack of the function.
  std::size_t params_idx = 0;
  for (int i = 0; i < kernel_args.size(); ++i) {
    int converter_idx = signature_metadata[i];
    if (converter_idx >= std::size(kExtractionInfos)) {
      throw py::value_error("corrupted signature metadata");
    }
    const ExtractionInfo &extraction_info = kExtractionInfos[converter_idx];
    if (extraction_info.size == 0) {
      continue; // skip adding constexpr parameters
    }
    config.params[params_idx] = alloca(extraction_info.size);
    if (!extraction_info.extractor(kernel_args[i], config.params[params_idx])) {
      return;
    }
    ++params_idx;
  }
  config.params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(global_scratch, config.params[params_idx])) {
    return;
  }

  if (!launchHook(launch_enter_hook, hook_args)) {
    return;
  }

  launchKernel(config);

  if (!launchHook(launch_exit_hook, hook_args)) {
    return;
  }
}

} // namespace

static std::map<std::string, int> getDeviceProperties(int device_id) {
  // Get device handle
  CUdevice device;
  cuDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  CUDA_CHECK(cuDeviceGetAttribute(
      &max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  CUDA_CHECK(cuDeviceGetAttribute(
      &max_num_regs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  CUDA_CHECK(cuDeviceGetAttribute(
      &multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CUDA_CHECK(
      cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  CUDA_CHECK(cuDeviceGetAttribute(&sm_clock_rate,
                                  CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  CUDA_CHECK(cuDeviceGetAttribute(
      &mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  CUDA_CHECK(cuDeviceGetAttribute(
      &mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  std::map<std::string, int> properties = {
      {"max_shared_mem", max_shared_mem},
      {"max_num_regs", max_num_regs},
      {"multiprocessor_count", multiprocessor_count},
      {"warpSize", warp_size},
      {"sm_clock_rate", sm_clock_rate},
      {"mem_clock_rate", mem_clock_rate},
      {"mem_bus_width", mem_bus_width},
  };
  return properties;
}

static std::tuple<uint64_t, uint64_t, int32_t, int32_t, int32_t>
loadBinary(char *name, char *data, int shared, CUdevice device) {
  CUfunction fun;
  CUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  int32_t n_max_threads = 0;
  // create driver handles
  CUcontext pctx = 0;

  py::gil_scoped_release release;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }

  CUDA_CHECK(cuModuleLoadData(&mod, data));
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  CUDA_CHECK(cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK(
      cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  CUDA_CHECK(cuFuncGetAttribute(&n_max_threads,
                                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));
  // set dynamic shared memory if necessary
  int shared_optin;
  CUDA_CHECK(cuDeviceGetAttribute(
      &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (shared > 49152 && shared_optin > 49152) {
    CUDA_CHECK(cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    CUDA_CHECK(cuDeviceGetAttribute(
        &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    CUDA_CHECK(cuFuncGetAttribute(&shared_static,
                                  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    CUDA_CHECK(
        cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_optin - shared_static));
  }

  return {(uint64_t)mod, (uint64_t)fun, n_regs, n_spills, n_max_threads};
}

typedef CUresult (*cuOccupancyMaxActiveClusters_t)(
    int *numClusters, CUfunction func, const CUlaunchConfig *config);

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    if (!libHandle) {                                                          \
      PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");      \
      return NULL;                                                             \
    }                                                                          \
    /* Clear any existing error */                                             \
    dlerror();                                                                 \
    symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
    /* Check for errors */                                                     \
    const char *err = dlerror();                                               \
    if (err) {                                                                 \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Failed to retrieve " #symbolName " from libcuda.so.1"); \
      dlclose(libHandle);                                                      \
      return NULL;                                                             \
    }                                                                          \
    return funcHandle;                                                         \
  }

defineGetFunctionHandle(getCuOccupancyMaxActiveClustersHandle,
                        cuOccupancyMaxActiveClusters);

defineGetFunctionHandle(getCuTensorMapEncodeTiledHandle,
                        cuTensorMapEncodeTiled);

static PyObject *occupancyMaxActiveClusters(PyObject *self, PyObject *args) {
  int clusterDimX = -1, clusterDimY = -1, clusterDimZ = -1,
      maxActiveClusters = -1;
  int shared = 0;
  CUfunction func;

  if (!PyArg_ParseTuple(args, "Kiiii", &func, &shared, &clusterDimX,
                        &clusterDimY, &clusterDimZ)) {
    return NULL;
  }

  // Let each SM have one block
  int maxActiveBlocks = 1;
  py::gil_scoped_release release;
  CUDA_CHECK(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));

  CUlaunchAttribute launchAttr[1];
  launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launchAttr[0].value.clusterDim.x = clusterDimX;
  launchAttr[0].value.clusterDim.y = clusterDimY;
  launchAttr[0].value.clusterDim.z = clusterDimZ;
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
  config.attrs = launchAttr;

  static cuOccupancyMaxActiveClusters_t cuOccupancyMaxActiveClusters = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuOccupancyMaxActiveClusters,
                                      getCuOccupancyMaxActiveClustersHandle);

  CUDA_CHECK(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  CUDA_CHECK(cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
  return PyLong_FromLong(maxActiveClusters);
}

static PyObject *setPrintfFifoSize(PyObject *self, PyObject *args) {
  long size;
  if (!PyArg_ParseTuple(args, "l", &size)) {
    return NULL;
  }
  if (size < 0) {
    PyErr_SetString(PyExc_ValueError, "fifo size must be non-negative");
    return NULL;
  }

  py::gil_scoped_release release;

  // Ensure we have an active context.
  CUcontext ctx = NULL;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, /*device=*/0));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
  }

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

  Py_INCREF(Py_None);
  return Py_None;
}

static void fillTMADescriptor(uint64_t desc_address, uint64_t global_address,
                              int swizzle, int elemSize, int elemType,
                              std::vector<int64_t> blockSize,
                              std::vector<int32_t> shape,
                              std::vector<int64_t> strides) {
  uint32_t blockSizeInt[5];
  uint64_t shapeInt[5];
  uint64_t stridesLL[5];

  int rank = blockSize.size();

  for (int i = 0; i < rank; ++i) {
    blockSizeInt[rank - i - 1] = blockSize[i];
  }

  if (rank != shape.size()) {
    throw py::value_error("Rank mismatch");
  }
  for (int i = 0; i < rank; ++i) {
    shapeInt[rank - i - 1] = shape[i];
  }

  if (rank != strides.size()) {
    throw py::value_error("Rank mismatch");
  }
  for (int i = 0; i + 1 < rank; ++i) {
    stridesLL[rank - i - 2] = elemSize * strides[i];
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);

  uint32_t elementStrides[5] = {1, 1, 1, 1, 1};
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK(cuTensorMapEncodeTiled(
      (CUtensorMap *)desc_address, (CUtensorMapDataType)elemType, rank,
      (void *)global_address, shapeInt, stridesLL, blockSizeInt, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, (CUtensorMapSwizzle)swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  return;
}

void init_triton_cuda_utils(py::module &&m) {
  m.def("load_binary", &loadBinary);
  m.def("get_device_properties", &getDeviceProperties);
  m.def("cuOccupancyMaxActiveClusters", &occupancyMaxActiveClusters);
  m.def("set_printf_fifo_size", &setPrintfFifoSize);
  m.def("fill_tma_descriptor", &fillTMADescriptor);
  m.def("build_signature_metadata", &buildSignatureMetadata);
  m.def("launch", &launch);
}
