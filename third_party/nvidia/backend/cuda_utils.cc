#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <alloca.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

#include "cuda.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "third_party/absl/cleanup/cleanup.h"

namespace {

struct UniquePyObjectDeleter {
  void operator()(PyObject *obj) { Py_DECREF(obj); }
};
// A unique_ptr for PyObjects that automatically calls Py_DECREF once it goes
// out of scope.
using UniquePyObjectPtr = std::unique_ptr<PyObject, UniquePyObjectDeleter>;

// Raise a python exception if the CUDA result code is not CUDA_SUCCESS.
// Can be called even on threads that do not hold Python's Global Interpreter
// Lock (GIL), as the function will acquire one if needed.
inline bool gpuAssert(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return true;
  const char *error = nullptr;
  cuGetErrorString(code, &error);
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyErr_Format(PyExc_RuntimeError, "Triton Error [CUDA]: %s", error);
  PyGILState_Release(gil_state);
  return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK(ans)                                                        \
  {{gpuAssert((ans), __FILE__, __LINE__);                                      \
  }                                                                            \
  }

#define CUDA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        return NULL;                                                           \
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
  Dim grid;            // Number of clusters per grid
  Dim cluster;         // Number of blocks per cluster
  int num_warps;       // number of warps per block
  int shared_memory;   // Size of shared memory in bytes to allocate
  CUstream stream;     // CUDA Stream on which to launch the kernel
  CUfunction function; // Pointer to the kernel to launch
  void **params;       // Parameters to pass to the kernel
};

// Launch a CUDA kernel with the given parameters. Raises a Python exception
// if the kernel launch fails.
PyObject *launchKernel(const TritonLaunchConfig &config) {
  // Launching the kernel might take a while and does not use Python APIs, so
  // we can release the Global Interpreter Lock so other threads can use Python
  // APIs if needed.
  Py_BEGIN_ALLOW_THREADS;
  const auto &grid = config.grid;
  const auto &cluster = config.cluster;
  if (grid.size() == 0) {
    PyEval_RestoreThread(_save);
    Py_RETURN_NONE;
  }
  if (cluster.size() == 1) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuLaunchKernel(
        config.function, grid.x, grid.y, grid.z, 32 * config.num_warps, 1, 1,
        config.shared_memory, config.stream, config.params, 0));
  } else {
    CUlaunchAttribute launchAttr[2];
    launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launchAttr[0].value.clusterDim.x = cluster.x;
    launchAttr[0].value.clusterDim.y = cluster.y;
    launchAttr[0].value.clusterDim.z = cluster.z;
    launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launchAttr[1].value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    CUlaunchConfig cu_config;
    cu_config.gridDimX = grid.x * cluster.x;
    cu_config.gridDimY = grid.y * cluster.y;
    cu_config.gridDimZ = grid.z * cluster.z;
    cu_config.blockDimX = 32 * config.num_warps;
    cu_config.blockDimY = 1;
    cu_config.blockDimZ = 1;
    cu_config.sharedMemBytes = config.shared_memory;
    cu_config.hStream = config.stream;
    cu_config.attrs = launchAttr;
    cu_config.numAttrs = 2;
    // cuLaunchKernelEx was added in CUDA 12, so load it dynamically to be
    // able to link on CUDA 11 and earlier.
    static cuLaunchKernelEx_t cuLaunchKernelExHandle =
        getLaunchKernelExHandle();
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuLaunchKernelExHandle(&cu_config, config.function, config.params, 0));
  }
  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}

// Interface used by various PyObject extractors to extract obj into a memory
// location pointed by ptr. Returns true if extraction succeeded, and false
// otherwise.
using ExtractorType = bool (*)(PyObject *obj, void *ptr);

// Enable peer access if dev_ptr is allocated on a different device than the
// device on which we will execute the kernel.
// Extract a CUDA device pointer from a pointer-like PyObject obj, and store
// it to the memory location pointed by ptr.
bool extractPointer(PyObject *obj, void *ptr) {
  auto dev_ptr = static_cast<CUdeviceptr *>(ptr);
  if (obj == Py_None) {
    *dev_ptr = static_cast<CUdeviceptr>(0); // valid nullptr
    return true;
  }
  if (PyLong_Check(obj)) {
    *dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return true;
  }
  UniquePyObjectPtr ret(PyObject_CallMethod(obj, "data_ptr", nullptr));
  if (!ret.get()) {
    PyErr_Format(PyExc_TypeError,
                 "Pointer argument must be either uint64 or have data_ptr "
                 "method, but got %R",
                 obj);
    return false;
  }
  if (!PyLong_Check(ret.get())) {
    PyErr_SetString(PyExc_TypeError,
                    "data_ptr method of Pointer object must return 64-bit int");
    return false;
  }
  *dev_ptr = PyLong_AsUnsignedLongLong(ret.get());
  if (PyErr_Occurred()) {
    return false;
  }
  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  CUresult status = cuPointerGetAttribute(
      dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
    PyErr_Format(PyExc_ValueError,
                 "Pointer argument cannot be accessed from Triton "
                 "(cpu tensor?)");
    return false;
  } else if (status != CUDA_SUCCESS) {
    CUDA_CHECK(status);
    return false;
  }
  return true;
}

// Extract a CUtensorMap descriptor from a python object, and store it to the
// memory location pointed by ptr.
bool extractTmaDesc(PyObject *obj, void *ptr) {
  if (sizeof(CUtensorMap *) != 8) {
    PyErr_SetString(PyExc_SystemError,
                    "extractTmaDesc() requires 64-bit compilation");
    return false;
  }

  UniquePyObjectPtr method_ret(
      PyObject_CallMethod(obj, "tma_desc_cpu_ptr", nullptr));
  // Checking the error retains context if tma_desc_cpu_ptr raises an exception.
  if (PyErr_Occurred()) {
    return false;
  }

  if (!method_ret) {
    PyErr_SetString(PyExc_SystemError, "Call to tma_desc_cpu_ptr() failed");
    return false;
  }

  if (!PyLong_Check(method_ret.get())) {
    PyErr_SetString(PyExc_TypeError,
                    "tma_desc_cpu_ptr() must return 64-bit int");
    return false;
  }

  uint64_t ptr_as_uint = PyLong_AsUnsignedLongLong(method_ret.get());
  if (PyErr_Occurred()) {
    return false;
  }

  if (!ptr_as_uint) {
    PyErr_SetString(PyExc_ValueError,
                    "received NULL ptr from tma_desc_cpu_ptr()");
    return false;
  }
  if (ptr_as_uint % 64 != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "tma_desc_cpu_ptr() must be 64-byte aligned");
    return false;
  }

  *static_cast<CUtensorMap *>(ptr) =
      *reinterpret_cast<CUtensorMap *>(ptr_as_uint);
  return true;
}

// For a given type T, maps to the Python API with signature `U (*)(PyObject*)`
// that can extract values of that type from a PyObject. Note that the return
// type U is not guaranteed to be the same as T, but it can be explicitly casted
// to T.
template <typename T, typename = void> constexpr auto kValueFunction = nullptr;
template <typename T>
constexpr auto
    kValueFunction<T, std::enable_if_t<std::is_integral_v<T> &&
                                       std::is_signed_v<T> && sizeof(T) <= 4>> =
        PyLong_AsLong;
template <> constexpr auto kValueFunction<std::int64_t> = PyLong_AsLongLong;
template <typename T>
constexpr auto kValueFunction<
    T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T> &&
                        sizeof(T) <= 4>> = PyLong_AsUnsignedLong;
template <>
constexpr auto kValueFunction<std::uint64_t> = PyLong_AsUnsignedLongLong;
template <typename T>
constexpr auto
    kValueFunction<T, std::enable_if_t<std::is_floating_point_v<T>>> =
        PyFloat_AsDouble;

// Extract a value of type T from obj and store it into memory pointed by ptr.
// Returns true if extraction succeeded, and false otherwise.
template <typename T> bool extractValue(PyObject *obj, void *ptr) {
  *static_cast<T *>(ptr) = static_cast<T>(kValueFunction<T>(obj));
  return PyErr_Occurred() == nullptr;
}

// Contains information necessary for extracting a certain type from a PyObject.
struct ExtractionInfo {
  // Prefixes of types reprs supported by the extractor.
  llvm::SmallVector<llvm::StringRef> supported_type_repr_prefixes;
  std::size_t size;        // Size required by the extracted value.
  ExtractorType extractor; // Function to call to extract the value.

  // Builds an ExtractionInfo for a given type T and a list of type reprs that
  // are backed by that type.
  template <typename T>
  static ExtractionInfo
  build(std::initializer_list<llvm::StringRef> supported_type_reprs,
        ExtractorType extractor = extractValue<T>) {
    return {supported_type_reprs, sizeof(T), extractor};
  }

  // Checks if the extractor supports extracting a given type repr.
  bool supports(llvm::StringRef type_repr) const {
    return llvm::any_of(
        supported_type_repr_prefixes,
        [&](llvm::StringRef prefix) { return type_repr.starts_with(prefix); });
  }
};

// Array of supported extractors
const ExtractionInfo kExtractionInfos[]{
    ExtractionInfo::build<std::int8_t>({"'i8'"}),
    ExtractionInfo::build<std::int16_t>({"'i16'"}),
    ExtractionInfo::build<std::int32_t>({"'i1'", "'i32'"}),
    ExtractionInfo::build<std::int64_t>({"'i64'"}),
    ExtractionInfo::build<std::uint8_t>({"'u8'"}),
    ExtractionInfo::build<std::uint16_t>({"'u16'"}),
    ExtractionInfo::build<std::uint32_t>({"'u1'", "'u32'"}),
    ExtractionInfo::build<std::uint64_t>({"'u64'"}),
    ExtractionInfo::build<float>({"'fp16'", "'bf16'", "'fp32'", "'f32'"}),
    ExtractionInfo::build<double>({"'fp64'"}),
    // Note: types are e.g. '*fp32', so no closing quote is intentional.
    ExtractionInfo::build<void *>({"'*"}, extractPointer),
    ExtractionInfo{
        {"None", "'none'"}, 0, nullptr}, // Represent constexprs as None
    ExtractionInfo::build<CUtensorMap>({"'nvTmaDesc'"}, extractTmaDesc),
};

// Finds an extractor that supports a given type_repr in the extractor list.
// Returns nullopt if no such extractor is found.
std::optional<char> findExtractor(llvm::StringRef type_repr) {
  constexpr std::size_t kNumExtractors = std::size(kExtractionInfos);
  static_assert(kNumExtractors < std::numeric_limits<char>::max(),
                "Not enough bits in a byte to store the extractor index");
  for (const auto &[idx, info] : llvm::enumerate(kExtractionInfos)) {
    if (info.supports(type_repr))
      return idx;
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
PyObject *buildSignatureMetadata(PyObject *self, PyObject *args) {
  PyObject *signature = nullptr;
  if (!PyArg_ParseTuple(args, "O", &signature)) {
    return nullptr;
  }
  if (!PyIter_Check(signature)) {
    PyErr_Format(PyExc_TypeError,
                 "expected signature to be an iterable, got %R", signature);
    return nullptr;
  }

  llvm::SmallVector<char, 16> signature_metadata;
  while (UniquePyObjectPtr obj_type{PyIter_Next(signature)}) {
    UniquePyObjectPtr repr(PyObject_Repr(obj_type.get()));
    if (!repr) {
      return nullptr;
    }
    UniquePyObjectPtr repr_str(
        PyUnicode_AsEncodedString(repr.get(), "utf-8", "~E~"));
    if (!repr_str) {
      return nullptr;
    }
    const char *repr_bytes = PyBytes_AsString(repr_str.get());
    if (!repr_bytes) {
      return nullptr;
    }
    std::optional<std::uint8_t> extractor_idx = findExtractor(repr_bytes);
    if (!extractor_idx.has_value()) {
      PyErr_Format(PyExc_TypeError,
                   "unexpected type %R in kernel signature, dir: %R",
                   obj_type.get(), PyObject_Dir(obj_type.get()));
      return nullptr;
    }
    signature_metadata.push_back(extractor_idx.value());
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }

  return PyBytes_FromStringAndSize(signature_metadata.data(),
                                   signature_metadata.size());
}

// Launch a Python callable hook with metadata passed as parameters.
bool launchHook(PyObject *hook, PyObject *metadata) {
  if (hook == Py_None) {
    return true;
  }
  UniquePyObjectPtr args(Py_BuildValue("(O)", metadata));
  if (!args) {
    return false;
  }
  UniquePyObjectPtr ret(PyObject_CallObject(hook, args.get()));
  return static_cast<bool>(ret);
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
PyObject *launch(PyObject *self, PyObject *args) {
  ensureCudaContext();
  TritonLaunchConfig config{};
  auto &grid = config.grid;
  auto &cluster = config.cluster;
  // PyObject* kernel_metadata = nullptr;
  PyObject *hook_args = nullptr;
  PyObject *launch_enter_hook = nullptr;
  PyObject *launch_exit_hook = nullptr;
  PyBytesObject *signature_metadata_bytes = nullptr;
  PyObject *kernel_args = nullptr;
  PyObject *global_scratch = nullptr;
  int num_ctas = 0;
  if (!PyArg_ParseTuple(args, "iiiKK(iiiiii)OOOSOO", &grid.x, &grid.y, &grid.z,
                        &config.stream, &config.function, &config.num_warps,
                        &num_ctas, &config.shared_memory, &cluster.x,
                        &cluster.y, &cluster.z, &hook_args, &launch_enter_hook,
                        &launch_exit_hook, &signature_metadata_bytes,
                        &global_scratch, &kernel_args)) {
    return nullptr;
  }
  if (num_ctas != cluster.size()) {
    PyErr_Format(
        PyExc_ValueError,
        "Expected cluster dimensions (%d, %d, %d) to have a total size of %d",
        cluster.x, cluster.y, cluster.z, num_ctas);
    return nullptr;
  }
  llvm::ArrayRef<char> signature_metadata(
      PyBytes_AS_STRING(signature_metadata_bytes),
      PyBytes_GET_SIZE(signature_metadata_bytes));
  UniquePyObjectPtr fast_kernel_args(PySequence_Fast(
      kernel_args, "Expected kernel_args to be a sequence or iterable"));
  if (!fast_kernel_args) {
    return nullptr;
  }
  llvm::ArrayRef<PyObject *> kernel_args_data(
      PySequence_Fast_ITEMS(fast_kernel_args.get()),
      PySequence_Fast_GET_SIZE(fast_kernel_args.get()));

  if (signature_metadata.size() != kernel_args_data.size()) {
    PyErr_Format(PyExc_TypeError,
                 "Expected kernel to have %d parameters, but got %d",
                 signature_metadata.size(), kernel_args_data.size());
    return nullptr;
  }

  // +1 for the global scratch pointer.
  std::size_t num_params = signature_metadata.size() + 1;
  // Use alloca to set up kernel parameters on the stack and avoid dynamic
  // memory allocations.
  config.params = static_cast<void **>(alloca(num_params * sizeof(void *)));
  // This loop has to stay in the same function that owns params, since we are
  // using alloca to allocate pointers to it on the stack of the function.
  std::size_t params_idx = 0;
  for (const auto &[converter_idx, arg] :
       llvm::zip(signature_metadata, kernel_args_data)) {
    if (converter_idx >= std::size(kExtractionInfos)) {
      PyErr_SetString(PyExc_ValueError, "corrupted signature metadata");
      return nullptr;
    }
    const ExtractionInfo &extraction_info = kExtractionInfos[converter_idx];
    if (extraction_info.size == 0) {
      continue; // skip adding constexpr parameters
    }
    config.params[params_idx] = alloca(extraction_info.size);
    if (!extraction_info.extractor(arg, config.params[params_idx])) {
      return nullptr;
    }
    ++params_idx;
  }
  config.params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(global_scratch, config.params[params_idx])) {
    return nullptr;
  }

  if (!launchHook(launch_enter_hook, hook_args)) {
    return nullptr;
  }

  if (!launchKernel(config)) {
    return nullptr;
  }

  if (!launchHook(launch_exit_hook, hook_args)) {
    return nullptr;
  }

  Py_RETURN_NONE;
}

} // namespace

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
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
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_num_regs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CUDA_CHECK_AND_RETURN_NULL(
      cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &sm_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  CUdevice device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }
  CUfunction fun;
  CUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  CUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGet(&device, 0));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(pctx));
  }

  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuModuleLoadData(&mod, data));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  // set dynamic shared memory if necessary
  int shared_optin;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
      &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (shared > 49152 && shared_optin > 49152) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
        &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
        &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_optin - shared_static));
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
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
  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
  Py_END_ALLOW_THREADS;

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

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
  Py_END_ALLOW_THREADS;
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

  Py_BEGIN_ALLOW_THREADS;

  // Ensure we have an active context.
  CUcontext ctx = NULL;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuDevicePrimaryCtxRetain(&ctx, /*device=*/0));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(ctx));
  }

  // We can't set the fifo size after running a kernel that calls printf.  This
  // is true even if the set() call is a nop and the new size is the same as the
  // old size.
  //
  // This is unfriendly, so check if the old size matches the new size, and skip
  // the set() call if so.
  size_t oldSize = 0;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuCtxGetLimit(&oldSize, CU_LIMIT_PRINTF_FIFO_SIZE));
  if (oldSize != size) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, size));
  }

  Py_END_ALLOW_THREADS;
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *fillTMADescriptor(PyObject *self, PyObject *args) {
  unsigned long long desc_address;
  unsigned long long global_address;
  int swizzle;
  int elemSize;
  int elemType;
  PyObject *blockSize;
  PyObject *shape;
  PyObject *strides;

  if (!PyArg_ParseTuple(args, "KKiiiOOO", &desc_address, &global_address,
                        &swizzle, &elemSize, &elemType, &blockSize, &shape,
                        &strides)) {
    return NULL;
  }

  PyObject *blockSizeFast = NULL;
  PyObject *shapeFast = NULL;
  PyObject *stridesFast = NULL;
  PyObject *result = NULL;

  auto cleanup = absl::MakeCleanup([&] {
    Py_XDECREF(blockSizeFast);
    Py_XDECREF(shapeFast);
    Py_XDECREF(stridesFast);
  });

  uint32_t blockSizeInt[5];
  uint64_t shapeInt[5];
  uint64_t stridesLL[5];

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    return result;
  int rank = PySequence_Fast_GET_SIZE(blockSizeFast);

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(blockSizeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "block size must be an int");
      return result;
    }
    blockSizeInt[rank - i - 1] = PyLong_AsLongLong(item);
  }

  shapeFast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shapeFast)
    return result;

  if (rank != PySequence_Fast_GET_SIZE(shapeFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Rank mismatch");
    return result;
  }
  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(shapeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      return result;
    }
    shapeInt[rank - i - 1] = PyLong_AsLong(item);
  }

  stridesFast = PySequence_Fast(strides, "strides must be a sequence");
  if (!stridesFast)
    return result;

  if (rank != PySequence_Fast_GET_SIZE(stridesFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Rank mismatch");
    return result;
  }
  for (int i = 0; i + 1 < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(stridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      return result;
    }
    stridesLL[rank - i - 2] = elemSize * PyLong_AsLongLong(item);
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);
  Py_DECREF(blockSizeFast);
  blockSizeFast = NULL;
  Py_DECREF(shapeFast);
  shapeFast = NULL;
  Py_DECREF(stridesFast);
  stridesFast = NULL;

  uint32_t elementStrides[5] = {1, 1, 1, 1, 1};
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUDA_CHECK_AND_RETURN_NULL(cuTensorMapEncodeTiled(
      (CUtensorMap *)desc_address, (CUtensorMapDataType)elemType, rank,
      (void *)global_address, shapeInt, stridesLL, blockSizeInt, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, (CUtensorMapSwizzle)swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  Py_RETURN_NONE;

  return result;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided cubin into CUDA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"cuOccupancyMaxActiveClusters", occupancyMaxActiveClusters, METH_VARARGS,
     "Python interface for cuOccupancyMaxActiveClusters function"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Python interface for cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, x), which "
     "controls how many bytes can be streamed from kernels before data starts "
     "being dropped.  This inherits all the limitations of this call; in "
     "particular it's an error to change this value after launching any kernel "
     "that calls printf()."},
    {"fill_tma_descriptor", fillTMADescriptor, METH_VARARGS, "doc"},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS,
     buildSignatureMetadata__doc__},
    {"launch", launch, METH_VARARGS, launch__doc__},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cuda_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cuda_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
