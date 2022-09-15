#include "triton/driver/error.h"
#include "triton/driver/llvm.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Target/PTX/PTXTranslation.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <Python.h>
#include <cctype>
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;
// namespace ir = triton::ir;
namespace drv = triton::driver;

using triton::cuGetInfo;

enum backend_t {
  HOST,
  CUDA,
  ROCM,
};

void cu_enable_peer_access(uint64_t peer_ptr) {
  CUcontext context;
  drv::dispatch::cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                       peer_ptr);
  try {
    drv::dispatch::cuCtxEnablePeerAccess(context, 0);
  } catch (drv::exception::cuda::peer_access_already_enabled) {
  }
}

void host_enqueue(uint64_t stream, uint64_t kernel, uint64_t grid_0,
                  uint64_t grid_1, uint64_t grid_2, uint64_t block_0,
                  uint64_t block_1, uint64_t block_2, void *args_ptr,
                  size_t args_size, int64_t shared_mem) {
  throw std::runtime_error("unsupported");
  // auto hst = kernel->module()->hst();
  // hst_->futures->reserve(hst_->futures->size() + grid[0]*grid[1]*grid[2]);
  // char* params = new char[args_size];
  // std::memcpy((void*)params, (void*)args, args_size);
  // for(size_t i = 0; i < grid[0]; i++)
  //   for(size_t j = 0; j < grid[1]; j++)
  //     for(size_t k = 0; k < grid[2]; k++)
  //       hst_->futures->emplace_back(hst_->pool->enqueue(hst->fn,
  //       (char**)params, int32_t(i), int32_t(j), int32_t(k)));
}

void cu_enqueue(uint64_t stream, uint64_t kernel, uint64_t grid_0,
                uint64_t grid_1, uint64_t grid_2, uint64_t block_0,
                uint64_t block_1, uint64_t block_2, void *args_ptr,
                size_t args_size, int64_t shared_mem) {
  void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, (void *)args_ptr,
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                    CU_LAUNCH_PARAM_END};
  drv::dispatch::cuLaunchKernel((CUfunction)kernel, grid_0, grid_1, grid_2,
                                block_0, block_1, block_2, shared_mem,
                                (CUstream)stream, nullptr, config);
}

long pow2_divisor(long N) {
  if (N % 16 == 0)
    return 16;
  if (N % 8 == 0)
    return 8;
  if (N % 4 == 0)
    return 4;
  if (N % 2 == 0)
    return 2;
  return 1;
}

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

// Returns something like "int16", whether dtype is a torch.dtype or
// triton.language.dtype.
std::string dtype_cache_key_part(const py::object &dtype) {
  if (py::hasattr(dtype, "cache_key_part")) {
    // Presumed to be a triton.language.dtype.
    return std::string(py::str(py::getattr(dtype, "cache_key_part")));
  } else {
    // Remove 'torch.' prefix from repr of torch.dtype.
    py::object repr = py::repr(dtype);
    size_t repr_len = PyUnicode_GET_LENGTH(repr.ptr());
    const char *repr_ptr = (const char *)PyUnicode_1BYTE_DATA(repr.ptr());
    if (repr_len <= 6 || strncmp(repr_ptr, "torch.", 6)) {
      throw std::logic_error("invalid dtype: " +
                             std::string(repr_ptr, repr_len));
    }
    return std::string(repr_ptr + 6, repr_len - 6);
  }
}

size_t get_pointer_range_size(uint64_t addr) {
  if (addr == 0)
    return 0;
  size_t size;
  drv::dispatch::cuPointerGetAttribute(&size, CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                                       (CUdeviceptr)addr);
  return size;
}

// Launch
void parse_args(py::list &args, py::list do_not_specialize,
                const std::string &func_key, py::list &arg_names,
                std::string &cache_key, std::string &params,
                size_t &params_size, py::dict constants, int num_warps,
                int num_stages) {
  size_t len = PyList_Size(args.ptr());
  params.reserve(8 * len); // 8 max bytes by argument
  char *params_ptr = &params[0];
  cache_key = func_key;
  cache_key += "-" + std::to_string(num_warps);
  cache_key += "-" + std::to_string(num_stages);
  cache_key += "-";
  for (int i = 0; i < len; i++) {
    cache_key += "_";
    py::int_ py_i = py::int_(i);
    bool specialize = !do_not_specialize.contains(py_i);
    py::object arg = args[i];
    auto arg_ptr = arg.ptr();

    // argument is `long`
    if (PyLong_Check(arg_ptr)) {
      int overflow;
      long long value = PyLong_AsLongLongAndOverflow(arg_ptr, &overflow);
      // values equal to 1 are specialized
      if (specialize && (value == 1)) {
        cache_key += "1";
        continue;
      }
      // int32, uint32, int64, and uint64 have different kernels
      if (!overflow && -0x8000'0000LL <= value && value <= 0x7FFF'FFFFLL) {
        cache_key += "int32";
        params_ptr = (char *)(((uintptr_t)params_ptr + 3) & (-4));
        std::memcpy(params_ptr, &value, 4);
        params_ptr += 4;
      } else if (!overflow && 0x8000'0000LL <= value &&
                 value <= 0xFFFF'FFFFLL) {
        cache_key += "uint32";
        params_ptr = (char *)(((uintptr_t)params_ptr + 3) & (-4));
        std::memcpy(params_ptr, &value, 4);
        params_ptr += 4;
      } else if (!overflow) {
        cache_key += "int64";
        params_ptr = (char *)(((uintptr_t)params_ptr + 7) & (-8));
        std::memcpy(params_ptr, &value, 8);
        params_ptr += 8;
      } else {
        if (PyErr_Occurred()) {
          throw std::logic_error("An error occurred?");
        }
        unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(arg_ptr);
        if (PyErr_Occurred()) {
          throw std::runtime_error("integer overflow in argument: " +
                                   std::string(py::str(arg)));
        }
        cache_key += "uint64";
        params_ptr = (char *)(((uintptr_t)params_ptr + 7) & (-8));
        std::memcpy(params_ptr, &unsigned_value, 8);
        params_ptr += 8;
      }
      if (!specialize)
        continue;
      // values divisible by small powers of 2 are specialized
      cache_key += "[multipleof(";
      cache_key += std::to_string(pow2_divisor(value));
      cache_key += ")]";
      continue;
    }
    // argument is `float`
    if (PyFloat_Check(arg_ptr)) {
      cache_key += "float32";
      float value = PyFloat_AsDouble(arg_ptr);
      params_ptr = (char *)(((uintptr_t)params_ptr + 3) & (-4));
      std::memcpy(params_ptr, &value, 4);
      params_ptr += 4;
      continue;
    }
    // argument is `bool`
    if (PyBool_Check(arg_ptr)) {
      cache_key += "bool";
      bool value = arg_ptr == Py_True ? true : false;
      std::memcpy(params_ptr, &value, 1);
      params_ptr += 1;
      continue;
    }
    // argument is tensor
    if (py::hasattr(arg, "data_ptr")) {
      py::object data_ptr = arg.attr("data_ptr")();
      long value = data_ptr.cast<long>();
      params_ptr = (char *)(((uintptr_t)params_ptr + 7) & (-8));
      // copy param
      std::memcpy(params_ptr, &value, 8);
      params_ptr += 8;
      // udpate cache key
      cache_key += dtype_cache_key_part(arg.attr("dtype"));
      cache_key += "*";
      cache_key += "[multipleof(";
      size_t range_size = get_pointer_range_size(value);
      cache_key += std::to_string(
          std::min(pow2_divisor(value), pow2_divisor(range_size)));
      cache_key += ")]";
      continue;
    }
    // argument is `constexpr`
    if (py::hasattr(arg, "value")) {
      py::object value = arg.attr("value");
      py::object name = arg_names[i];
      constants[name] = value;
      py::object repr = py::repr(value);
      const char *start = (const char *)PyUnicode_1BYTE_DATA(repr.ptr());
      size_t len = PyUnicode_GET_LENGTH(repr.ptr());
      cache_key += std::string(start, len);
      continue;
    }
    std::string ty_str =
        arg.attr("__class__").attr("__name__").cast<std::string>();
    if (ty_str == "NoneType") {
      cache_key += "None";
      continue;
    }
    std::string err_msg = "Received type '" + ty_str + "' for argument " +
                          std::to_string(i) + "." +
                          " Only int, float, bool, torch.Tensor, and "
                          "triton.language.constexpr are supported.";
    throw std::runtime_error(err_msg);
  }
  params_size = (std::ptrdiff_t)(params_ptr - &params[0]);
}

void parse_args(py::list &args, py::list &arg_names, std::string &params,
                size_t &params_size, py::dict constants) {
  size_t len = PyList_Size(args.ptr());
  params.reserve(8 * len); // 8 max bytes by argument
  char *params_ptr = params.data();
  for (int i = 0; i < len; i++) {
    py::object arg = args[i];
    auto arg_ptr = arg.ptr();

    if (PyLong_Check(arg_ptr)) {
      int overflow{};
      long long value = PyLong_AsLongLongAndOverflow(arg_ptr, &overflow);

      if (!overflow && -0x8000'0000LL <= value && value <= 0x7FFF'FFFFLL) {
        params_ptr = (char *)(((uintptr_t)params_ptr + 3) & (-4));
        std::memcpy(params_ptr, &value, 4);
        params_ptr += 4;
      } else if (!overflow && 0x8000'0000LL <= value &&
                 value <= 0xFFFF'FFFFLL) {
        params_ptr = (char *)(((uintptr_t)params_ptr + 3) & (-4));
        std::memcpy(params_ptr, &value, 4);
        params_ptr += 4;
      } else if (!overflow) {
        params_ptr = (char *)(((uintptr_t)params_ptr + 7) & (-8));
        std::memcpy(params_ptr, &value, 8);
        params_ptr += 8;
      } else {
        if (PyErr_Occurred()) {
          throw std::logic_error("An error occurred?");
        }
        unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(arg_ptr);
        if (PyErr_Occurred()) {
          throw std::runtime_error("integer overflow in argument: " +
                                   std::string(py::str(arg)));
        }
        params_ptr = (char *)(((uintptr_t)params_ptr + 7) & (-8));
        std::memcpy(params_ptr, &unsigned_value, 8);
        params_ptr += 8;
      }
      continue;
    }

    if (PyFloat_Check(arg_ptr)) {
      float value = PyFloat_AsDouble(arg_ptr);
      params_ptr = (char *)(((uintptr_t)params_ptr + 3) & (-4));
      std::memcpy(params_ptr, &value, 4);
      params_ptr += 4;
      continue;
    }

    // argument is `bool`
    if (PyBool_Check(arg_ptr)) {
      bool value = arg_ptr == Py_True ? true : false;
      std::memcpy(params_ptr, &value, 1);
      params_ptr += 1;
      continue;
    }
    // argument is torch.tensor, get data_ptr as memory address
    if (py::hasattr(arg, "data_ptr")) {
      py::object data_ptr = arg.attr("data_ptr")();
      long value = data_ptr.cast<long>();
      params_ptr = (char *)(((uintptr_t)params_ptr + 7) & (-8));
      // copy param
      std::memcpy(params_ptr, &value, 8);
      params_ptr += 8;
      // udpate cache key
      continue;
    }
    // argument is `constexpr`
    if (py::hasattr(arg, "value")) {
      py::object value = arg.attr("value");
      py::object name = arg_names[i];
      constants[name] = value;
      continue;
    }
    // argument is `LoadedBinary`
    if (py::hasattr(arg, "get_sass")) {
      // Do nothing, just a placeholder here to indicate validity.
      continue;
    }

    std::string ty_str =
        arg.attr("__class__").attr("__name__").cast<std::string>();
    std::string err_msg = "Received type '" + ty_str + "' for argument " +
                          std::to_string(i) + "." +
                          " Only int, float, bool, torch.Tensor, and "
                          "triton.language.constexpr are supported.";
    throw std::runtime_error(err_msg);
  }

  params_size = (std::ptrdiff_t)(params_ptr - &params[0]);
}

void init_triton_runtime(py::module &&m) {
  // wrap backend_t
  py::enum_<backend_t>(m, "backend")
      .value("HOST", HOST)
      .value("CUDA", CUDA)
      // .value("ROCM", ROCM)
      .export_values();

  // enable peer-to-peer
  m.def("enable_peer_access", [](backend_t backend, uint64_t peer_ptr) {
    if (backend != CUDA)
      throw std::runtime_error("P2P only supported on CUDA devices!");
    cu_enable_peer_access(peer_ptr);
  });

  // get range size for the given pointer
  m.def("get_pointer_range_size", &get_pointer_range_size);

  // cache key
  m.def("launch", [](py::list args, py::list do_not_specialize,
                     const std::string &func_key, py::list &arg_names,
                     py::object device, py::int_ stream, py::dict bin_cache,
                     py::int_ num_warps, py::int_ num_stages,
                     py::function add_to_cache, py::object grid) {
    // parse arguments to compute cache key, compile-time constants and packed
    // kernel arguments
    long _num_warps = PyLong_AsLong(num_warps.ptr());
    long _num_stages = PyLong_AsLong(num_stages.ptr());
    std::string cache_key;
    std::string params;
    size_t params_size;
    py::dict constants;
    parse_args(args, do_not_specialize, func_key, arg_names, cache_key, params,
               params_size, constants, _num_warps, _num_stages);

    // get cached binary
    py::str key(cache_key);
    py::bool_ noop = false;
    if (!bin_cache.contains(key)) {
      noop = add_to_cache(key, args, device, num_warps, num_stages);
    }
    if (noop)
      return (py::object)py::none();
    py::object bin = bin_cache[key];

    // get grid
    py::sequence seq;
    if (!PySequence_Check(grid.ptr()))
      seq = grid(constants);
    else
      seq = grid;
    int size = seq.size();
    int grid_0 = py::cast<int>(seq[0]);
    int grid_1 = size < 2 ? 1 : py::cast<int>(seq[1]);
    int grid_2 = size < 3 ? 1 : py::cast<int>(seq[2]);

    // enqueue
    uint64_t kernel = py::cast<uint64_t>(bin.attr("kernel"));
    uint64_t shared_mem = py::cast<uint64_t>(bin.attr("shared_mem"));

    // actually launch
    void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, params.data(),
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &params_size,
                      CU_LAUNCH_PARAM_END};
    uint64_t _stream = PyLong_AsLong(stream.ptr());
    if (grid_0 * grid_1 * grid_2 > 0) {
      // release the gil in case the enqueue blocks
      // cuda will block if too many ops are enqueued
      py::gil_scoped_release allow_threads;
      drv::dispatch::cuLaunchKernel((CUfunction)kernel, grid_0, grid_1, grid_2,
                                    _num_warps * 32, 1, 1, shared_mem,
                                    (CUstream)_stream, nullptr, config);
    }
    return bin;
  });

  m.def("cc", [](backend_t backend, uint64_t device) -> int {
    if (backend == CUDA) {
      CUdevice dev = (CUdevice)device;
      int major = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(dev);
      int minor = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(dev);
      return major * 10 + minor;
    }
    return -1;
  });

  m.def("launch_binary", [](py::object binary, py::list args,
                            py::list do_not_specialize, py::list arg_names,
                            py::int_ stream, py::int_ num_warps,
                            py::int_ num_stages, py::object grid) {
    long _num_warps = PyLong_AsLong(num_warps.ptr());
    long _num_stages = PyLong_AsLong(num_stages.ptr());

    // get grid
    py::sequence seq;
    py::dict constants;
    std::string params;
    size_t params_size{};
    parse_args(args, arg_names, params, params_size, constants);
    if (!PySequence_Check(grid.ptr()))
      seq = grid(constants);
    else
      seq = grid;

    int size = seq.size();
    int grid_0 = py::cast<int>(seq[0]);
    int grid_1 = size < 2 ? 1 : py::cast<int>(seq[1]);
    int grid_2 = size < 3 ? 1 : py::cast<int>(seq[2]);

    uint64_t kernel = py::cast<uint64_t>(binary.attr("kernel"));
    uint64_t shared_mem = py::cast<uint64_t>(binary.attr("shared_mem"));

    // actually launch
    void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, params.data(),
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &params_size,
                      CU_LAUNCH_PARAM_END};
    uint64_t _stream = PyLong_AsLong(stream.ptr());
    const int numGrids = grid_0 * grid_1 * grid_2;
    if (numGrids) {
      // release the gil in case the enqueue blocks
      // cuda will block if too many ops are enqueued
      py::gil_scoped_release allow_threads;
      drv::dispatch::cuLaunchKernel((CUfunction)kernel, grid_0, grid_1, grid_2,
                                    _num_warps * 32, 1, 1, shared_mem,
                                    (CUstream)_stream, nullptr, config);
    }
    return binary;
  });

  // query maximum shared memory
  m.def("max_shared_memory", [](backend_t backend, uint64_t device) {
    if (backend == HOST)
      return 0;
    if (backend == CUDA)
      return cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN>(
          device);
    return -1;
  });

  // query DRAM & L2 cache
  m.def("memory_clock_rate", [](backend_t backend, uint64_t device) {
    if (backend == CUDA)
      return cuGetInfo<CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE>(device);
    return -1;
  });
  m.def("global_memory_bus_width", [](backend_t backend, uint64_t device) {
    if (backend == CUDA)
      return cuGetInfo<CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH>(device);
    return -1;
  });
  m.def("l2_cache_size", [](backend_t backend, uint64_t device) {
    if (backend == CUDA)
      return cuGetInfo<CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE>(device);
    return -1;
  });

  // query clock rate (in kilohertz)
  m.def("clock_rate", [](backend_t backend, uint64_t device) {
    if (backend == CUDA)
      return cuGetInfo<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>(device);
    return -1;
  });

  m.def("num_sm", [](backend_t backend, uint64_t device) {
    if (backend == CUDA)
      return cuGetInfo<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>(device);
    return -1;
  });

  // enqueue
  m.def("enqueue",
        [](backend_t backend, uint64_t stream, uint64_t kernel, uint64_t grid_0,
           uint64_t grid_1, uint64_t grid_2, uint64_t block_0, uint64_t block_1,
           uint64_t block_2, const std::string &args, int64_t shared_mem) {
          void *args_ptr = (void *)args.data();
          size_t args_size = args.size();
          // release the gil in case the enqueue blocks
          // cuda will block if too many ops are enqueued
          py::gil_scoped_release allow_threads;
          if (backend == HOST)
            host_enqueue(stream, kernel, grid_0, grid_1, grid_2, block_0,
                         block_1, block_2, args_ptr, args_size, shared_mem);
          if (backend == CUDA)
            cu_enqueue(stream, kernel, grid_0, grid_1, grid_2, block_0, block_1,
                       block_2, args_ptr, args_size, shared_mem);
        });
}

/*****************************************************************************/
/* Python bindings for triton::codegen                                       */
/*****************************************************************************/
typedef std::map<std::string, py::object> asm_map_t;

// ---------------------------------------
// Load provided assembly code into driver
// ---------------------------------------

// CUDA
std::tuple<uint64_t, uint64_t> cu_load_binary(const std::string &name,
                                              asm_map_t &asm_map,
                                              size_t n_shared_bytes,
                                              uint64_t dev) {
  // load assembly
  std::string assembly;
  if (asm_map.find("cubin") != asm_map.end())
    assembly = py::cast<std::string>(asm_map["cubin"]);
  else
    assembly = py::cast<std::string>(asm_map["ptx"]);
  // create driver handles
  CUfunction fun;
  CUmodule mod;
  drv::dispatch::cuModuleLoadData(&mod, assembly.c_str());
  drv::dispatch::cuModuleGetFunction(&fun, mod, name.c_str());
  // set dynamic shared memory if necessary
  int shared_optin;
  drv::dispatch::cuDeviceGetAttribute(
      &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      dev);
  if (n_shared_bytes > 49152 && shared_optin > 49152) {
    drv::dispatch::cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED);
    int shared_total, shared_static;
    int n_spills, n_reg;
    drv::dispatch::cuDeviceGetAttribute(
        &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        dev);
    drv::dispatch::cuFuncGetAttribute(&shared_static,
                                      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun);
    drv::dispatch::cuFuncGetAttribute(&n_spills,
                                      CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);
    drv::dispatch::cuFuncGetAttribute(&n_reg, CU_FUNC_ATTRIBUTE_NUM_REGS, fun);
    drv::dispatch::cuFuncSetAttribute(
        fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_optin - shared_static);
  }
  return std::make_tuple((uint64_t)mod, (uint64_t)fun);
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<mlir::triton::CacheModifier>(m, "CACHE_MODIFIER")
      .value("NONE", mlir::triton::CacheModifier::NONE)
      .value("CA", mlir::triton::CacheModifier::CA)
      .value("CG", mlir::triton::CacheModifier::CG)
      .export_values();

  py::enum_<mlir::triton::EvictionPolicy>(m, "EVICTION_POLICY")
      .value("NORMAL", mlir::triton::EvictionPolicy::NORMAL)
      .value("EVICT_FIRST", mlir::triton::EvictionPolicy::EVICT_FIRST)
      .value("EVICT_LAST", mlir::triton::EvictionPolicy::EVICT_LAST)
      .export_values();

  py::enum_<mlir::triton::RedOp>(m, "REDUCE_OP")
      .value("ADD", mlir::triton::RedOp::ADD)
      .value("FADD", mlir::triton::RedOp::FADD)
      .value("MIN", mlir::triton::RedOp::MIN)
      .value("MAX", mlir::triton::RedOp::MAX)
      .value("FMIN", mlir::triton::RedOp::FMIN)
      .value("FMAX", mlir::triton::RedOp::FMAX)
      .value("XOR", mlir::triton::RedOp::XOR);

  py::enum_<mlir::triton::RMWOp>(m, "ATOMIC_OP")
      .value("ADD", mlir::triton::RMWOp::ADD)
      .value("FADD", mlir::triton::RMWOp::FADD)
      .value("AND", mlir::triton::RMWOp::AND)
      .value("OR", mlir::triton::RMWOp::OR)
      .value("XOR", mlir::triton::RMWOp::XOR)
      // .value("XCHG", mlir::triton::RMWOp::Xchg)
      .value("MAX", mlir::triton::RMWOp::MAX)
      .value("MIN", mlir::triton::RMWOp::MIN)
      .value("UMIN", mlir::triton::RMWOp::UMIN)
      .value("UMAX", mlir::triton::RMWOp::UMAX);

  py::class_<mlir::MLIRContext>(m, "context")
      .def(py::init<>())
      .def("load_triton", [](mlir::MLIRContext &self) {
        self.getOrLoadDialect<mlir::triton::TritonDialect>();
      });
  // .def(py::init([](){
  //   mlir::MLIRContext context;
  //   context.getOrLoadDialect<mlir::triton.TritonDialect>();
  //   // TODO: should we return a (raw/unique) pointer here?
  //   return context;
  // }));

  // py::class_<ir::value>(m, "value")
  //     .def("multiple_of", [](ir::value *self, int val) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self)) {
  //         instr->set_metadata(ir::metadata::multiple_of, val);
  //       } else
  //         throw std::runtime_error("multiple_of");
  //     })
  //     .def("max_contiguous", [](ir::value *self, int val) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self)) {
  //         instr->set_metadata(ir::metadata::max_contiguous, val);
  //       } else
  //         throw std::runtime_error("max_contiguous");
  //     })
  //     .def("set_fdiv_ieee_rounding", [](ir::value *self, bool val) {
  //       if (auto *instr = dynamic_cast<ir::binary_operator*>(self))
  //         instr->set_fdiv_ieee_rounding(val);
  //       else
  //         throw std::runtime_error("set_fdiv_ieee_rounding");
  //     })
  //     .def("ops", [](ir::value *self) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self)) {
  //         return instr->ops();
  //       }
  //       throw std::runtime_error("cannot use ops()");
  //     })
  //     .def("replace_all_uses_with", &ir::value::replace_all_uses_with)
  //     .def("erase_from_parent", [](ir::value *self) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self))
  //         return instr->erase_from_parent();
  //       throw std::runtime_error("cannot use erase_from_parent");
  //     })
  //     .def_property("name", &ir::value::get_name, &ir::value::set_name)
  //     .def_property_readonly("type", &ir::value::get_type);

  // // // Do we need under in TritonIR ?
  // // py::class_<ir::undef_value, ir::constant>(m, "undef")
  // //     .def("get", &ir::undef_value::get, ret::reference);

  py::class_<mlir::Type>(m, "type")
      .def("is_integer", &mlir::Type::isInteger)
      .def("is_fp16", &mlir::Type::isF16);

  py::class_<mlir::Value>(m, "value")
      .def("set_attr",
           [](mlir::Value &self, std::string &name,
              mlir::Attribute &attr) -> void {
             if (mlir::Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               /* issue an warning */
             }
           });
  py::class_<mlir::BlockArgument, mlir::Value>(m, "block_arguement");

  py::class_<mlir::Region>(m, "region")
      .def("get_parent_region", &mlir::Region::getParentRegion, ret::reference)
      .def("size", [](mlir::Region &self) { return self.getBlocks().size(); })
      .def("empty", &mlir::Region::empty);

  py::class_<mlir::Block>(m, "block")
      .def("arg",
           [](mlir::Block &self, int index) -> mlir::BlockArgument {
             return self.getArgument(index);
           })
      .def("get_num_arguments", &mlir::Block::getNumArguments)
      .def("dump", &mlir::Block::dump)
      .def("move_before", &mlir::Block::moveBefore)
      .def("insert_before", &mlir::Block::insertBefore)
      .def("get_parent", &mlir::Block::getParent, ret::reference)
      .def("merge_block_before",
           [](mlir::Block &self, mlir::Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.end(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with", [](mlir::Block &self, mlir::Value &v,
                                           mlir::Value &newVal) {
        v.replaceUsesWithIf(newVal, [&](mlir::OpOperand &operand) {
          mlir::Operation *user = operand.getOwner();
          mlir::Block *currentBlock = user->getBlock();
          while (currentBlock) {
            if (currentBlock == &self)
              return true;
            // Move up one level
            currentBlock = currentBlock->getParent()->getParentOp()->getBlock();
          }
          return false;
        });
      });

  // using eattr = ir::attribute_kind_t;
  // py::enum_<eattr>(m, "attribute_kind")
  //     .value("readonly", eattr::readonly)
  //     .value("writeonly", eattr::writeonly)
  //     .value("noalias", eattr::noalias)
  //     .value("aligned", eattr::aligned)
  //     .value("multiple_of", eattr::multiple_of)
  //     .value("retune", eattr::retune)
  //     .value("not_implemented", eattr::not_implemented);

  py::class_<mlir::Attribute>(m, "attribute");
  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "integer_attr");
  py::class_<mlir::BoolAttr, mlir::Attribute>(m, "bool_attr");

  // Ops
  py::class_<mlir::OpState>(m, "OpState")
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("str",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self->print(os);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });
  // scf Ops
  py::class_<mlir::scf::ForOp, mlir::OpState>(m, "ForOp");
  py::class_<mlir::scf::IfOp, mlir::OpState>(m, "IfOp")
      .def("get_then_block", &mlir::scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &mlir::scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &mlir::scf::IfOp::thenYield)
      .def("get_else_yield", &mlir::scf::IfOp::elseYield);
  py::class_<mlir::scf::YieldOp, mlir::OpState>(m, "YieldOp");
  py::class_<mlir::scf::WhileOp, mlir::OpState>(m, "WhileOp")
      .def("get_before", &mlir::scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &mlir::scf::WhileOp::getAfter, ret::reference);
  py::class_<mlir::scf::ConditionOp, mlir::OpState>(m, "CondtionOp");

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("str",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> mlir::FuncOp {
             return self.lookupSymbol<mlir::FuncOp>(funcName);
           });

  py::class_<mlir::FuncOp, mlir::OpState>(m, "function")
      // .def_property_readonly("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](mlir::FuncOp &self, unsigned idx) -> mlir::BlockArgument {
             return self.getArgument(idx);
           })
      .def(
          "add_entry_block",
          [](mlir::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](mlir::FuncOp &self, int arg_no, const std::string &name, int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def("reset_type", &mlir::FuncOp::setType);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint");

  py::class_<mlir::OpBuilder>(m, "builder", py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      // // getters
      .def_property_readonly("context", &mlir::OpBuilder::getContext,
                             ret::reference)
      .def("create_module",
           [](mlir::OpBuilder &self) -> mlir::ModuleOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::ModuleOp>(loc);
           })
      .def("ret",
           [](mlir::OpBuilder &self, std::vector<mlir::Value> &vals) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::ReturnOp>(loc, vals);
           })
      .def("call",
           [](mlir::OpBuilder &self, mlir::FuncOp &func,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::CallOp>(loc, func, args);
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](mlir::OpBuilder &self, mlir::Block &block) -> void {
             self.setInsertionPointToStart(&block);
           })
      .def("set_insertion_point_to_end",
           [](mlir::OpBuilder &self, mlir::Block &block) {
             self.setInsertionPointToEnd(&block);
           })
      .def(
          "get_insertion_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return self.getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point", &mlir::OpBuilder::saveInsertionPoint)
      .def("restore_insertion_point", &mlir::OpBuilder::restoreInsertionPoint)
      // .def("set_insert_point", [](ir::builder *self,
      // std::pair<ir::basic_block*, ir::instruction*> pt) {
      //   ir::basic_block *bb = pt.first;
      //   ir::instruction *instr = pt.second;
      //   if (instr) {
      //     if (bb != instr->get_parent())
      //       throw std::runtime_error("invalid insertion point, instr not in
      //       bb");
      //     self->set_insert_point(instr);
      //   } else {
      //     assert(bb);
      //     self->set_insert_point(bb);
      //   }
      // })
      // Attr
      .def("get_bool_attr", &mlir::OpBuilder::getBoolAttr)
      .def("get_int32_attr", &mlir::OpBuilder::getI32IntegerAttr)
      // Use arith.ConstantOp to create constants
      // // Constants
      // .def("get_int1", &ir::builder::get_int1, ret::reference)
      .def("get_int32",
           [](mlir::OpBuilder &self, int64_t v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI32Type()));
           })
      // .def("get_uint32", &ir::builder::get_int32, ret::reference)
      // .def("get_int64", [](ir::builder *self, int64_t v) { return
      // self->get_int64((uint64_t)v); }, ret::reference) .def("get_uint64",
      // &ir::builder::get_int64, ret::reference) .def("get_float16",
      // &ir::builder::get_float16, ret::reference)
      .def("get_float32",
           [](mlir::OpBuilder &self, float v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ConstantOp>(
                 loc, self.getF32FloatAttr(v));
           })
      .def("get_null_value",
           [](mlir::OpBuilder &self, mlir::Type &type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             if (type.isa<mlir::FloatType>())
               return self.create<mlir::arith::ConstantOp>(
                   loc, self.getF32FloatAttr(0.0));
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getNoneType();
           })
      .def("get_int1_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type { return self.getI8Type(); })
      .def("get_int16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::IntegerType>(16);
           })
      .def(
          "get_int32_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI32Type(); })
      .def(
          "get_int64_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI64Type(); })
      .def("get_fp8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::triton::Float8Type>();
           })
      .def("get_bf8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::triton::BFloat8Type>();
           })
      .def(
          "get_half_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF16Type(); })
      .def("get_bf16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getBF16Type();
           })
      .def(
          "get_float_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF32Type(); })
      .def(
          "get_double_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF64Type(); })
      .def("get_ptr_ty",
           [](mlir::OpBuilder &self, mlir::Type &type,
              int addrSpace) -> mlir::Type {
             return mlir::triton::PointerType::get(type, addrSpace);
           })
      .def("get_block_ty",
           [](mlir::OpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
             return mlir::RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
             return self.getFunctionType(inTypes, outTypes);
           })

      // Ops
      .def("create_function",
           [](mlir::OpBuilder &self, std::string name,
              mlir::Type &funcType) -> mlir::FuncOp {
             // TODO: loc
             auto loc = self.getUnknownLoc();
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               return self.create<mlir::FuncOp>(loc, name, funcTy);
             }
             throw std::runtime_error("invalid function type");
           })
      .def("get_or_insert_function",
           [](mlir::OpBuilder &self, mlir::ModuleOp &module,
              std::string &funcName, mlir::Type &funcType) -> mlir::FuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<mlir::FuncOp>(funcOperation);
             auto loc = self.getUnknownLoc();
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               return self.create<mlir::FuncOp>(loc, funcName, funcTy);
             }
             throw std::runtime_error("invalid function type");
           })
      .def(
          "create_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            mlir::Region *parent = self.getBlock()->getParent();
            return self.createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](mlir::OpBuilder &self, mlir::Region &parent,
             std::vector<mlir::Type> &argTypes) -> mlir::Block * {
            auto argLoc = self.getUnknownLoc();
            llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(),
                                                         argLoc);
            return self.createBlock(&parent, {}, argTypes, argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return new mlir::Block();
          },
          ret::reference)
      // Structured control flow
      .def("create_for_op",
           [](mlir::OpBuilder &self, mlir::Value &lb, mlir::Value &ub,
              mlir::Value &step,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::ForOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ForOp>(loc, lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              mlir::Value &condition, bool withElse) -> mlir::scf::IfOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::IfOp>(loc, retTypes, condition,
                                                 withElse);
           })
      .def("create_yield_op",
           [](mlir::OpBuilder &self,
              std::vector<mlir::Value> &yields) -> mlir::scf::YieldOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::YieldOp>(loc, yields);
           })
      .def("create_while_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::WhileOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::WhileOp>(loc, retTypes, initArgs);
           })
      .def("create_condtion_op",
           [](mlir::OpBuilder &self, mlir::Value &cond,
              std::vector<mlir::Value> &args) -> mlir::scf::ConditionOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ConditionOp>(loc, cond, args);
           })

      // miscellious
      .def("create_make_range",
           [](mlir::OpBuilder &self, int start, int end) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto retType =
                 mlir::RankedTensorType::get({end - start}, self.getI32Type());
             return self.create<mlir::triton::MakeRangeOp>(loc, retType, start,
                                                           end);
           })
      .def("create_get_program_id",
           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::GetProgramIdOp>(
                 loc, self.getI32Type(), axis);
           })

      // Cast instructions
      .def("create_bitcast",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::BitcastOp>(loc, dstType, src);
           })
      // .def("create_cast", &ir::builder::create_cast)
      // .def("create_ptr_to_int", &ir::builder::create_ptr_to_int)
      .def("create_si_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::SIToFPOp>(loc, dstType, src);
           })
      .def("create_ui_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::UIToFPOp>(loc, dstType, src);
           })
      .def("create_fp_to_si",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::FPToSIOp>(loc, dstType, src);
           })
      .def("create_fp_to_ui",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::FPToUIOp>(loc, dstType, src);
           })
      .def("create_fp_ext",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ExtFOp>(loc, dstType, src);
           })
      .def("create_fp_trunc",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::TruncFOp>(loc, dstType, src);
           })
      // .def("create_int_cast", &ir::builder::create_int_cast)
      // .def("create_downcast", &ir::builder::create_downcast)
      .def("create_to_index",
           [](mlir::OpBuilder &self, mlir::Value &input) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::IndexCastOp>(loc, input,
                                                          self.getIndexType());
           })

      .def("create_fmul",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::MulFOp>(loc, lhs, rhs);
           })
      .def("create_fdiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivFOp>(loc, lhs, rhs);
           })
      .def("create_frem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemFOp>(loc, lhs, rhs);
           })
      .def("create_fadd",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AddFOp>(loc, lhs, rhs);
           })
      .def("create_fsub",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::SubFOp>(loc, lhs, rhs);
           })
      .def("create_mul",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::MulIOp>(loc, lhs, rhs);
           })
      .def("create_sdiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
           })
      .def("create_udiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
           })
      .def("create_srem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
           })
      .def("create_urem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemUIOp>(loc, lhs, rhs);
           })
      .def("create_add",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AddIOp>(loc, lhs, rhs);
           })
      .def("create_sub",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::SubIOp>(loc, lhs, rhs));
           })
      .def("create_shl",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShLIOp>(loc, lhs, rhs));
           })
      .def("create_lshr",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShRUIOp>(loc, lhs, rhs));
           })
      .def("create_ashr",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShRSIOp>(loc, lhs, rhs));
           })
      // AddPtr (similar to GEP)
      .def("create_addptr",
           [](mlir::OpBuilder &self, mlir::Value &ptr,
              mlir::Value &offset) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::AddPtrOp>(loc, ptr.getType(), ptr,
                                                        offset);
           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
           })
      .def("create_icmpSLT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
           })
      .def("create_icmpSGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
           })
      .def("create_icmpSGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
           })
      .def("create_icmpULE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ule, lhs, rhs);
           })
      .def("create_icmpULT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ult, lhs, rhs);
           })
      .def("create_icmpUGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::uge, lhs, rhs);
           })
      .def("create_icmpUGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
           })
      .def("create_icmpEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
           })
      .def("create_icmpNE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
           })
      .def("create_fcmpOGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
           })
      .def("create_fcmpOLE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
           })
      .def("create_fcmpOGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
           })
      .def("create_fcmpOEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
           })
      .def("create_fcmpONE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
           })
      .def("create_fcmpULT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ULT, lhs, rhs);
           })
      .def("create_fcmpUGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UGT, lhs, rhs);
           })
      .def("create_fcmpULE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ULE, lhs, rhs);
           })
      .def("create_fcmpUGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UGE, lhs, rhs);
           })
      .def("create_fcmpUEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UEQ, lhs, rhs);
           })
      .def("create_fcmpUNE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UNE, lhs, rhs);
           })
      // // Logical
      .def("create_and",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AndIOp>(loc, lhs, rhs);
           })
      .def("create_xor",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
           })
      .def("create_or",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::OrIOp>(loc, lhs, rhs);
           })
      // // Input/Output
      .def("create_load",
           [](mlir::OpBuilder &self, mlir::Value &ptrs,
              mlir::triton::CacheModifier cacheModifer,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::LoadOp>(
                 loc, ptrs, cacheModifer, evictionPolicy, isVolatile);
           })
      .def("create_store",
           [](mlir::OpBuilder &self, mlir::Value &ptrs,
              mlir::Value &value) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::triton::StoreOp>(loc, ptrs, value);
           })
      .def("create_masked_load",
           [](mlir::OpBuilder &self, mlir::Value &ptrs, mlir::Value &mask,
              std::optional<mlir::Value> &other,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::LoadOp>(
                 loc, ptrs, mask, other.value_or(mlir::Value()), cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_masked_store",
           [](mlir::OpBuilder &self, mlir::Value &ptrs, mlir::Value &val,
              mlir::Value &mask) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::triton::StoreOp>(loc, ptrs, val, mask);
           })
      .def("create_view",
           [](mlir::OpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto argType = arg.getType()
                                .dyn_cast<mlir::RankedTensorType>()
                                .getElementType();
             return self.create<mlir::triton::ViewOp>(
                 loc, mlir::RankedTensorType::get(shape, argType), arg);
           })
      .def(
          "create_expand_dims",
          [](mlir::OpBuilder &self, mlir::Value &arg, int axis) -> mlir::Value {
            auto loc = self.getUnknownLoc();
            auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
            auto argEltType = argType.getElementType();
            std::vector<int64_t> retShape = argType.getShape();
            retShape.insert(retShape.begin() + axis, 1);
            return self.create<mlir::triton::ExpandDimsOp>(
                loc, mlir::RankedTensorType::get(retShape, argEltType), arg,
                axis);
          })
      .def("create_cat",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto lhsType = lhs.getType().dyn_cast<mlir::RankedTensorType>();
             auto rhsType = rhs.getType().dyn_cast<mlir::RankedTensorType>();
             if (!(lhsType.getShape().size() == 1 &&
                   rhsType.getShape().size() == 1))
               throw std::runtime_error(
                   "shape not supported by cat. Expecting rank-1 inputs");
             std::vector<int64_t> shape{lhsType.getShape()[0] +
                                        rhsType.getShape()[0]};
             return self.create<mlir::triton::CatOp>(
                 loc,
                 mlir::RankedTensorType::get(shape, lhsType.getElementType()),
                 lhs, rhs);
           })
      .def("create_broadcast",
           [](mlir::OpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             if (auto argType =
                     arg.getType().dyn_cast<mlir::RankedTensorType>())
               return self.createOrFold<mlir::triton::BroadcastOp>(
                   loc,
                   mlir::RankedTensorType::get(shape, argType.getElementType()),
                   arg);
             throw std::runtime_error(
                 "arg is not of RankedTensorType, use create_splat");
           })
      .def("create_splat",
           [](mlir::OpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto argType = arg.getType();
             auto ret = self.createOrFold<mlir::triton::SplatOp>(
                 loc, mlir::RankedTensorType::get(shape, argType), arg);
             return ret;
           })
      // // atomic
      .def("create_atomic_cas",
           [](mlir::OpBuilder &self, mlir::Value &ptr, mlir::Value &cmp,
              mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto ptrType = ptr.getType().dyn_cast<mlir::triton::PointerType>();
             mlir::Type dstType = ptrType.getPointeeType();
             return self.create<mlir::triton::AtomicCASOp>(loc, dstType, ptr,
                                                           cmp, val);
           })
      .def("create_atomic_rmw",
           [](mlir::OpBuilder &self, mlir::triton::RMWOp rmwOp,
              mlir::Value &ptr, mlir::Value &val,
              mlir::Value &mask) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto ptrType = ptr.getType().dyn_cast<mlir::triton::PointerType>();
             mlir::Type dstType = ptrType.getPointeeType();
             return self.create<mlir::triton::AtomicRMWOp>(loc, dstType, rmwOp,
                                                           ptr, val, mask);
           })
      // External
      .def("create_external_elementwise",
           [](mlir::OpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<mlir::Value> &argList,
              mlir::Type retType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::ExtElemwiseOp>(
                 loc, retType, argList, libName, libPath, symbol);
           })
      // Built-in instruction
      .def("create_get_program_id",
           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::GetProgramIdOp>(
                 loc, self.getI32Type(), self.getI32IntegerAttr(axis));
           })
      .def("create_get_num_programs",
           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::GetNumProgramsOp>(
                 loc, self.getI32Type(), self.getI32IntegerAttr(axis));
           })
      .def("create_dot",
           [](mlir::OpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, bool allowTF32) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::DotOp>(loc, c.getType(), a, b, c,
                                                     allowTF32);
           })
      .def("create_exp",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::ExpOp>(loc, val);
           })
      .def("create_cos",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::CosOp>(loc, val);
           })
      .def("create_sin",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::SinOp>(loc, val);
           })
      .def("create_log",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::LogOp>(loc, val);
           })
      .def("create_sqrt",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::SqrtOp>(loc, val);
           })
      // .def("create_trans", &ir::builder::create_trans, ret::reference)
      .def("create_reduce",
           [](mlir::OpBuilder &self, mlir::Value &operand,
              mlir::triton::RedOp redOp, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto inputTensorType =
                 operand.getType().dyn_cast<mlir::RankedTensorType>();
             std::vector<int64_t> shape = inputTensorType.getShape();
             shape.erase(shape.begin() + axis);
             auto resType = mlir::RankedTensorType::get(
                 shape, inputTensorType.getElementType());
             return self.create<mlir::triton::ReduceOp>(loc, resType, redOp,
                                                        operand, axis);
           })
      .def("create_select",
           [](mlir::OpBuilder &self, mlir::Value &condition,
              mlir::Value &trueValue, mlir::Value &falseValue) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::SelectOp>(loc, condition, trueValue,
                                                falseValue);
           })
      // // Intrinsics
      // // These have no place in the IR, and hopefully they can be removed at
      // some point .def("create_umulhi", &ir::builder::create_umulhi,
      // ret::reference) .def("create_barrier", &ir::builder::create_barrier,
      // ret::reference);
      ;

  py::class_<mlir::PassManager>(m, "pass_manager")
      .def(py::init<mlir::MLIRContext *>())
      .def("enable_debug",
           [](mlir::PassManager &self) {
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.elideLargeElementsAttrs(16);
             self.enableIRPrinting(
                 /*shouldPrintBeforePass=*/nullptr,
                 /*shouldPrintAfterPass=*/
                 [](mlir::Pass *pass, mlir::Operation *) {
                   return getBoolEnv("MLIR_ENABLE_DUMP");
                 },
                 /*printModuleScope=*/false,
                 /*printAfterOnlyOnChange=*/true,
                 /*printAfterOnlyOnFailure*/ false, llvm::dbgs(),
                 printingFlags);
           })
      .def("run",
           [](mlir::PassManager &self, mlir::ModuleOp &mod) -> bool {
             return mlir::succeeded(self.run(mod.getOperation()));
           })
      .def(
          "add_sccp_pass",
          [](mlir::PassManager &self) { self.addPass(mlir::createSCCPPass()); })
      .def("add_coalesce_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUCoalescePass());
           })
      .def("add_symbol_dce_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createSymbolDCEPass());
           })
      .def("add_inliner_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createInlinerPass());
           })
      .def("add_canonicalizer_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createCanonicalizerPass());
           })
      .def("add_cse_pass",
           [](mlir::PassManager &self) { self.addPass(mlir::createCSEPass()); })
      .def("add_triton_combine_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::triton::createCombineOpsPass());
           })
      .def("add_convert_triton_to_tritongpu_pass",
           [](mlir::PassManager &self, int numWarps) {
             self.addPass(
                 mlir::triton::createConvertTritonToTritonGPUPass(numWarps));
           })
      .def("add_tritongpu_pipeline_pass",
           [](mlir::PassManager &self, int numStages) {
             self.addPass(mlir::createTritonGPUPipelinePass(numStages));
           })
      .def("add_triton_gpu_combine_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUCombineOpsPass());
           })
      .def("add_triton_gpu_verifier_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUVerifier());
           })
      .def("add_triton_gpu_to_llvm", [](mlir::PassManager &self) {
        self.addPass(mlir::triton::createConvertTritonGPUToLLVMPass());
      });
}

void init_triton_translation(py::module &m) {
  m.def("translate_triton_gpu_to_llvmir", [](mlir::ModuleOp op) -> std::string {
    llvm::LLVMContext llvmContext;
    auto llvmModule =
        ::mlir::triton::translateTritonGPUToLLVMIR(&llvmContext, op);

    std::string str;
    llvm::raw_string_ostream os(str);
    llvmModule->print(os, nullptr);
    os.flush();
    return str;
  });

  m.def("translate_triton_gpu_to_ptx",
        [](mlir::ModuleOp module, uint64_t device)
            -> std::tuple<std::string /*ptx code*/, size_t /*shem size*/> {
          auto [ptxCode, cc, version, ptxasPath] =
              triton::translateTritonGPUToPTX(module, device);

          mlir::PassManager pm(module->getContext());
          auto pass = std::make_unique<mlir::Allocation>(module);
          size_t size = pass->getSharedMemorySize();

          return std::make_tuple(ptxCode, size);
        });

  m.def("compile_ptx_to_cubin",
        [](const std::string &ptxCode, uint64_t device) -> py::object {
          py::gil_scoped_release allow_threads;
          int version;
          int cc;
          std::string ptxasPath;
          triton::getCuCCAndVersionFromDevice(device, &cc, &version,
                                              &ptxasPath);

          std::string cubin = drv::ptx_to_cubin(ptxCode, ptxasPath, cc);
          py::bytes bytes(cubin);
          return bytes;
        });

  m.def(
      "load_binary",
      [](backend_t backend, const std::string &name, asm_map_t &asm_map,
         size_t n_shared_bytes, uint64_t dev) {
        py::gil_scoped_release allow_threads;
        assert(backend == CUDA); // Only CUDA is supported now.
        return cu_load_binary(name, asm_map, n_shared_bytes, dev);
      },
      py::return_value_policy::take_ownership);
}

void init_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  // init_triton_codegen(std::move(subm.def_submodule("code_gen")));
  init_triton_runtime(std::move(subm.def_submodule("runtime")));
  init_triton_ir(std::move(subm.def_submodule("ir")));
  init_triton_translation(subm);
}
