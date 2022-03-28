#include "triton/codegen/pass.h"
#include "triton/codegen/target.h"
#include "triton/driver/error.h"
#include "triton/driver/llvm.h"
#include "triton/ir/builder.h"
#include "triton/ir/dispatch.h"
#include "triton/ir/enums.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/print.h"
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Python.h"
#include <regex>
#include <sstream>
#include <string>
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"

namespace py = pybind11;
namespace ir = triton::ir;
namespace drv = triton::driver;


/*****************************************************************************/
/* Python bindings for triton::driver                                        */
/*****************************************************************************/
// information query
template<CUdevice_attribute attr>
int cuGetInfo(CUdevice device) {
  int res;
  drv::dispatch::cuDeviceGetAttribute(&res, attr, device);
  return res;
}

template<hipDeviceAttribute_t attr>
int hipGetInfo(hipDevice_t device) {
  int res;
  drv::dispatch::hipDeviceGetAttribute(&res, attr, device);
  return res;
}

enum backend_t {
  HOST,
  CUDA,
  ROCM,
};

void cu_enable_peer_access(uint64_t peer_ptr){
  CUcontext context;
  drv::dispatch::cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, peer_ptr);
  try {
      drv::dispatch::cuCtxEnablePeerAccess(context, 0);
  } catch (drv::exception::cuda::peer_access_already_enabled) {}
}

void host_enqueue(uint64_t stream, uint64_t kernel,
                  uint64_t grid_0, uint64_t grid_1, uint64_t grid_2,
                  uint64_t block_0, uint64_t block_1, uint64_t block_2,
                  void* args_ptr, size_t args_size, int64_t shared_mem){
  throw std::runtime_error("unsupported");
// auto hst = kernel->module()->hst();
// hst_->futures->reserve(hst_->futures->size() + grid[0]*grid[1]*grid[2]);
// char* params = new char[args_size];
// std::memcpy((void*)params, (void*)args, args_size);
// for(size_t i = 0; i < grid[0]; i++)
//   for(size_t j = 0; j < grid[1]; j++)
//     for(size_t k = 0; k < grid[2]; k++)
//       hst_->futures->emplace_back(hst_->pool->enqueue(hst->fn, (char**)params, int32_t(i), int32_t(j), int32_t(k)));
}

void cu_enqueue(uint64_t stream, uint64_t kernel,
                uint64_t grid_0, uint64_t grid_1, uint64_t grid_2,
                uint64_t block_0, uint64_t block_1, uint64_t block_2,
                void* args_ptr, size_t args_size, int64_t shared_mem){
  void *config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER, (void*)args_ptr,
      CU_LAUNCH_PARAM_BUFFER_SIZE,    &args_size,
      CU_LAUNCH_PARAM_END
  };
  drv::dispatch::cuLaunchKernel((CUfunction)kernel, grid_0, grid_1, grid_2, 
                                block_0, block_1, block_2, 
                                shared_mem, (CUstream)stream, nullptr, config);
}

void hip_enqueue(uint64_t stream, uint64_t kernel,
                uint64_t grid_0, uint64_t grid_1, uint64_t grid_2,
                uint64_t block_0, uint64_t block_1, uint64_t block_2,
                void* args_ptr, size_t args_size, int64_t shared_mem) {
  void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, (void*)args_ptr,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,    &args_size,
      HIP_LAUNCH_PARAM_END
  };
  drv::dispatch::hipModuleLaunchKernel((hipFunction_t)kernel, grid_0, grid_1, grid_2, 
                                block_0, block_1, block_2, 
                                shared_mem, (hipStream_t)stream, nullptr, config);

}

long pow2_divisor(long N){
    if(N % 16 == 0) return 16;
    if(N % 8 == 0) return 8;
    if(N % 4 == 0) return 4;
    if(N % 2 == 0) return 2;
    return 1;
}

// Returns something like "int16", whether dtype is a torch.dtype or
// triton.language.dtype.
std::string dtype_cache_key_part(const py::object& dtype) {
  if (py::hasattr(dtype, "cache_key_part")) {
    // Presumed to be a triton.language.dtype.
    return std::string(py::str(py::getattr(dtype, "cache_key_part")));
  } else {
    // Remove 'torch.' prefix from repr of torch.dtype.
    py::object repr = py::repr(dtype);
    size_t repr_len = PyUnicode_GET_LENGTH(repr.ptr());
    const char* repr_ptr = (const char*)PyUnicode_1BYTE_DATA(repr.ptr());
    if (repr_len <= 6 || strncmp(repr_ptr, "torch.", 6)) {
      throw std::logic_error("invalid dtype: " + std::string(repr_ptr, repr_len));
    }
    return std::string(repr_ptr + 6, repr_len - 6);
  }
}

size_t get_pointer_range_size(uint64_t addr){
  if(addr == 0)
    return 0;
  size_t size;
  drv::dispatch::cuPointerGetAttribute(&size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)addr);
  return size;
}

// Launch
void parse_args(py::list& args, py::list do_not_specialize, const std::string& func_key, py::list& arg_names,
                std::string& cache_key, std::string& params, size_t& params_size, py::dict constants,
                int num_warps, int num_stages) {
    size_t len = PyList_Size(args.ptr());
    params.reserve(8*len); // 8 max bytes by argument
    char* params_ptr = &params[0];
    cache_key = func_key;
    cache_key += "-" + std::to_string(num_warps);
    cache_key += "-" + std::to_string(num_stages);
    cache_key += "-";
    for(int i = 0; i < len; i++){
      cache_key += "_";
      py::int_ py_i = py::int_(i);
      bool specialize = !do_not_specialize.contains(py_i);
      py::object arg = args[i];
      auto arg_ptr = arg.ptr();

      // argument is `long`
      if(PyLong_Check(arg_ptr)){
        int overflow;
        long long value = PyLong_AsLongLongAndOverflow(arg_ptr, &overflow);
        // values equal to 1 are specialized
        if(specialize && (value == 1)){
          cache_key += "1";
          continue;
        }
        // int32, uint32, int64, and uint64 have different kernels
        if (!overflow && -0x8000'0000LL <= value && value <= 0x7FFF'FFFFLL) {
          cache_key += "int32";
          params_ptr = (char*)(((uintptr_t)params_ptr + 3) & (-4));
          std::memcpy(params_ptr, &value, 4);
          params_ptr += 4;
        } else if (!overflow && 0x8000'0000LL <= value && value <= 0xFFFF'FFFFLL) {
          cache_key += "uint32";
          params_ptr = (char*)(((uintptr_t)params_ptr + 3) & (-4));
          std::memcpy(params_ptr, &value, 4);
          params_ptr += 4;
        } else if (!overflow) {
          cache_key += "int64";
          params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
          std::memcpy(params_ptr, &value, 8);
          params_ptr += 8;
        } else {
          if (PyErr_Occurred()) {
            throw std::logic_error("An error occurred?");
          }
          unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(arg_ptr);
          if (PyErr_Occurred()) {
            throw std::runtime_error("integer overflow in argument: " + std::string(py::str(arg)));
          }
          cache_key += "uint64";
          params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
          std::memcpy(params_ptr, &unsigned_value, 8);
          params_ptr += 8;
        }
        if(!specialize)
          continue;
        // values divisible by small powers of 2 are specialized
        cache_key += "[multipleof(";
        cache_key += std::to_string(pow2_divisor(value));
        cache_key += ")]";
        continue;
      }
      // argument is `float`
      if(PyFloat_Check(arg_ptr)){
        cache_key += "float32";
        float value = PyFloat_AsDouble(arg_ptr);
        params_ptr = (char*)(((uintptr_t)params_ptr + 3) & (-4));
        std::memcpy(params_ptr, &value, 4);
        params_ptr += 4;
        continue;
      }
      // argument is `bool`
      if(PyBool_Check(arg_ptr)){
        cache_key += "bool";
        bool value =  arg_ptr == Py_True ? true : false;
        std::memcpy(params_ptr, &value, 1);
        params_ptr += 1;
        continue;
      }
      // argument is tensor
      if(py::hasattr(arg, "data_ptr")){
        py::object data_ptr = arg.attr("data_ptr")();
        long value = data_ptr.cast<long>();
        params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
        // copy param
        std::memcpy(params_ptr, &value, 8);
        params_ptr += 8;
        // udpate cache key
        cache_key += dtype_cache_key_part(arg.attr("dtype"));
        cache_key += "*";
        cache_key += "[multipleof(";
        size_t range_size = get_pointer_range_size(value);
        cache_key += std::to_string(std::min(pow2_divisor(value), pow2_divisor(range_size)));
        cache_key += ")]";
        continue;
      }
      // argument is `constexpr`
      if(py::hasattr(arg, "value")){
        py::object value = arg.attr("value");
        py::object name = arg_names[i];
        constants[name] = value;
        py::object repr = py::repr(value);
        const char* start = (const char*)PyUnicode_1BYTE_DATA(repr.ptr());
        size_t len = PyUnicode_GET_LENGTH(repr.ptr());
        cache_key += std::string(start, len);
        continue;
      }
      std::string ty_str = arg.attr("__class__").attr("__name__").cast<std::string>();
      if(ty_str == "NoneType"){
        cache_key += "None";
        continue;
      }
      std::string err_msg = "Received type '" + ty_str + "' for argument " + std::to_string(i) + "."
                            + " Only int, float, bool, torch.Tensor, and triton.language.constexpr are supported.";
      throw std::runtime_error(err_msg);
    }
  params_size = (std::ptrdiff_t)(params_ptr - &params[0]);
}

//

void init_triton_runtime(py::module &&m) {

  // m.def("current_stream", [](uint64_t device){
  //   return (uint64_t)(c10::cuda::getCurrentCUDAStream(device).stream());
  // });

  // wrap backend_t
  py::enum_<backend_t>(m, "backend")
    .value("HOST", HOST)
    .value("CUDA", CUDA)
    .value("ROCM", ROCM)
    .export_values();

  // enable peer-to-peer
  m.def("enable_peer_access", [](backend_t backend, uint64_t peer_ptr) {
      if (backend != CUDA)
        throw std::runtime_error("P2P only supported on CUDA devices!");
      cu_enable_peer_access(peer_ptr);
    }
  );

  // get range size for the given pointer
  m.def("get_pointer_range_size", &get_pointer_range_size);


  // cache key
  m.def("launch", [](py::list args, py::list do_not_specialize, const std::string& func_key, py::list& arg_names, 
                     py::object device, py::int_ stream, py::dict bin_cache, py::int_ num_warps, py::int_ num_stages, 
                     py::function add_to_cache, py::object grid){
    // parse arguments to compute cache key, compile-time constants and packed kernel arguments
    long _num_warps = PyLong_AsLong(num_warps.ptr());
    long _num_stages = PyLong_AsLong(num_stages.ptr());
    std::string cache_key;
    std::string params;
    size_t params_size;
    py::dict constants;
    parse_args(args, do_not_specialize, func_key, arg_names, cache_key, params, params_size, constants, _num_warps, _num_stages);

    // get cached binary
    py::str key(cache_key);
    py::bool_ noop = false;
    if(!bin_cache.contains(key)) {
      noop = add_to_cache(key, args, device, num_warps, num_stages);
    }
    if (noop)
      return (py::object)py::none();
    py::object bin = bin_cache[key];

    // get grid
    py::sequence seq;
    if(!PySequence_Check(grid.ptr()))
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
    void *config[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, params.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE, &params_size,
        CU_LAUNCH_PARAM_END
    };
    uint64_t _stream = PyLong_AsLong(stream.ptr());
    if(grid_0*grid_1*grid_2 > 0) {
      // release the gil in case the enqueue blocks
      // cuda will block if too many ops are enqueued
      py::gil_scoped_release allow_threads;
      drv::dispatch::cuLaunchKernel((CUfunction)kernel, grid_0, grid_1, grid_2, 
                                    _num_warps*32, 1, 1, shared_mem, (CUstream)_stream, 
                                     nullptr, config);
   }
    return bin;
  });

  m.def("cc", [](backend_t backend, uint64_t device) -> int {
    if (backend == CUDA) {
      CUdevice dev = (CUdevice)device;
      int major = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(dev);
      int minor = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(dev);
      return major*10 + minor;
    }
    return -1;
  });

  // query maximum shared memory
  m.def("max_shared_memory", [](backend_t backend, uint64_t device) {
      if (backend == HOST)
        return 0;
      if(backend == CUDA) 
        return cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN>(device);
      if(backend == ROCM)
        return hipGetInfo<hipDeviceAttributeMaxSharedMemoryPerBlock>(device);
      return -1;
  });

  // query DRAM & L2 cache
  m.def("memory_clock_rate", [](backend_t backend, uint64_t device) {
    if (backend == CUDA) return cuGetInfo<CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE>(device);
    return -1;
  });
  m.def("global_memory_bus_width", [](backend_t backend, uint64_t device) {
    if (backend == CUDA) return cuGetInfo<CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH>(device);
    return -1;
  });
  m.def("l2_cache_size", [](backend_t backend, uint64_t device) {
    if (backend == CUDA) return cuGetInfo<CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE>(device);
    return -1;
  });

  // query clock rate (in kilohertz)
  m.def("clock_rate", [](backend_t backend, uint64_t device) {
    if (backend == CUDA) return cuGetInfo<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>(device);
    return -1;
  });

  m.def("num_sm", [](backend_t backend, uint64_t device) {
    if (backend == CUDA) return cuGetInfo<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>(device);
    return -1;
  });

  // enqueue
  m.def("enqueue", [](backend_t backend, uint64_t stream, uint64_t kernel,
                      uint64_t grid_0, uint64_t grid_1, uint64_t grid_2,
                      uint64_t block_0, uint64_t block_1, uint64_t block_2,
                      const std::string &args, int64_t shared_mem){
    void* args_ptr = (void*)args.data();
    size_t args_size = args.size();
    // release the gil in case the enqueue blocks
    // cuda will block if too many ops are enqueued
    py::gil_scoped_release allow_threads;
    if(backend == HOST)
      host_enqueue(stream, kernel, grid_0, grid_1, grid_2, block_0, block_1, block_2, args_ptr, args_size, shared_mem);
    if(backend == CUDA)
      cu_enqueue(stream, kernel, grid_0, grid_1, grid_2, block_0, block_1, block_2, args_ptr, args_size, shared_mem);
    if(backend == ROCM)
      hip_enqueue(stream, kernel, grid_0, grid_1, grid_2, block_0, block_1, block_2, args_ptr, args_size, shared_mem);
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
std::tuple<uint64_t, uint64_t> cu_load_binary(const std::string& name, asm_map_t &asm_map, size_t n_shared_bytes, uint64_t dev){
  // load assembly
  std::string assembly;
  if(asm_map.find("cubin") != asm_map.end())
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
  drv::dispatch::cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
  if(n_shared_bytes > 49152 && shared_optin > 49152){
    drv::dispatch::cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED);
    int shared_total, shared_static;
    int n_spills, n_reg;
    drv::dispatch::cuDeviceGetAttribute(&shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
    drv::dispatch::cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun);
    drv::dispatch::cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,  fun);
    drv::dispatch::cuFuncGetAttribute(&n_reg, CU_FUNC_ATTRIBUTE_NUM_REGS, fun);
    drv::dispatch::cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin - shared_static);
  }
  return std::make_tuple((uint64_t)mod, (uint64_t)fun);
}

// ROCM
std::tuple<uint64_t, uint64_t> hip_load_binary(const std::string& name, asm_map_t &asm_map, size_t n_shared_bytes, uint64_t dev){
  py::bytes _assembly = asm_map["hsaco"];
  std::string assembly = py::cast<std::string>(_assembly);
  // HSA-CO -> hipModule
  hipModule_t mod = drv::amdgpu_to_hipmodule(assembly);
  // Handle to the kernel
  hipFunction_t fun;
  drv::dispatch::hipModuleGetFunction(&fun, mod, name.c_str());
  // record asm
  return std::make_tuple((uint64_t)mod, (uint64_t)fun);
}

// --------------------------------------- 
// Compile Triton-IR to assembly
// --------------------------------------- 

// CUDA
std::tuple<std::string, asm_map_t, int> cu_compile_ttir(const std::string& name, ir::module &ir, 
                                                               uint64_t device, int num_warps, int num_stages,
                                                               asm_map_t &asm_map){

  int n_shared_bytes;
  py::gil_scoped_release allow_threads;
  llvm::LLVMContext ctx;
  // device properties
  CUdevice dev = (CUdevice)device;
  size_t major = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(dev);
  size_t minor = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(dev);
  size_t cc = major*10 + minor;
  int version;
  std::string ptxas_path = drv::path_to_ptxas(version);
  // Triton-IR -> NVPTX LLVM-IR
  triton::codegen::nvidia_cu_target target(cc);
  auto llvm = triton::codegen::add_passes_to_emit_bin(ir, ctx, &target, cc, num_warps, num_stages, n_shared_bytes);
  std::string tmp;
  llvm::raw_string_ostream llir(tmp);
  llir << *llvm;
  llir.flush();
  asm_map["llir"] = py::cast(tmp);
  // LLVM-IR -> PTX
  std::string ptx = drv::llir_to_ptx(llvm.get(), cc, version);
  asm_map["ptx"] = py::cast(ptx);
  // PTX -> Binary
  std::string cubin = drv::ptx_to_cubin(ptx, ptxas_path, cc);
  if(!cubin.empty()){
    py::bytes bytes(cubin);
    asm_map["cubin"] = bytes;
  }
  return std::make_tuple(name, asm_map, n_shared_bytes);
}

// HIP
std::tuple<std::string, asm_map_t, int> hip_compile_ttir(const std::string& name, ir::module &ir, 
                                                                uint64_t device, int num_warps, int num_stages, 
                                                                asm_map_t &asm_map){
  llvm::LLVMContext ctx;
  // Triton-IR -> NVPTX LLVM-IR
  triton::codegen::amd_cl_target target;
  int n_shared_bytes;
  auto llvm = triton::codegen::add_passes_to_emit_bin(ir, ctx, &target, 70, num_warps, num_stages, n_shared_bytes);
  std::string tmp;
  llvm::raw_string_ostream llir(tmp);
  llir << *llvm;
  llir.flush();
  asm_map["llir"] = py::cast(tmp);
  // LLVM-IR -> HSA-CO
  std::string path = drv::llir_to_amdgpu(llvm.get(), "gfx908");
  asm_map["hsaco"] = py::cast(path);
  return std::make_tuple(name, asm_map, n_shared_bytes);
}

void init_triton_codegen(py::module &&m) {
  m.def(
      "compile_ttir", [](backend_t backend, ir::module &ir, uint64_t device, int num_warps, int num_stages) {
        std::string name = ir.get_function_list()[0]->get_name();
        // record asm as we generate
        asm_map_t asm_map;
        std::ostringstream ttir;
        ir.print(ttir);
        asm_map["ttir"] = py::cast(ttir.str());
        llvm::LLVMContext ctx;
        if(backend == CUDA)
          return cu_compile_ttir(name, ir, device, num_warps, num_stages, asm_map);
        if(backend == ROCM)
          return hip_compile_ttir(name, ir, device, num_warps, num_stages, asm_map);
      }, py::return_value_policy::take_ownership);
  m.def("load_binary", [](backend_t backend, const std::string& name, asm_map_t &asm_map, size_t n_shared_bytes, uint64_t dev){
	py::gil_scoped_release allow_threads;
        if(backend == CUDA)
          return cu_load_binary(name, asm_map, n_shared_bytes, dev);
        if(backend == ROCM)
          return hip_load_binary(name, asm_map, n_shared_bytes, dev);
      }, py::return_value_policy::take_ownership);
}

/*****************************************************************************/
/* User-facing language features                                             */
/*****************************************************************************/

void init_triton_frontend(py::module &&m) {
  using ret = py::return_value_policy;

  // programming model
  m.def("program_id", &ir::dispatch::program_id, ret::reference);
  m.def("num_programs", &ir::dispatch::num_programs, ret::reference);
  // binary
  m.def("add", &ir::dispatch::add, ret::reference);
  m.def("sub", &ir::dispatch::sub, ret::reference);
  m.def("mul", &ir::dispatch::mul, ret::reference);
  m.def("truediv", &ir::dispatch::truediv, ret::reference);
  m.def("floordiv", &ir::dispatch::floordiv, ret::reference);
  m.def("fdiv", &ir::dispatch::fdiv, ret::reference);
  m.def("mod", &ir::dispatch::mod, ret::reference);
  m.def("and_", &ir::dispatch::and_, ret::reference);
  m.def("or_", &ir::dispatch::or_, ret::reference);
  m.def("xor_", &ir::dispatch::xor_, ret::reference);
  m.def("lshr", &ir::dispatch::lshr, ret::reference);
  m.def("shl", &ir::dispatch::shl, ret::reference);
  // unary
  m.def("plus", &ir::dispatch::plus, ret::reference);
  m.def("minus", &ir::dispatch::minus, ret::reference);
  m.def("invert", &ir::dispatch::invert, ret::reference);
  // comparison
  m.def("greater_than", &ir::dispatch::greater_than, ret::reference);
  m.def("greater_equal", &ir::dispatch::greater_equal, ret::reference);
  m.def("less_than", &ir::dispatch::less_than, ret::reference);
  m.def("less_equal", &ir::dispatch::less_equal, ret::reference);
  m.def("equal", &ir::dispatch::equal, ret::reference);
  m.def("not_equal", &ir::dispatch::not_equal, ret::reference);
  // block creation
  m.def("arange", &ir::dispatch::arange, ret::reference);
  m.def("zeros", &ir::dispatch::zeros, ret::reference);
  // type manipuatation
  m.def("cat", &ir::dispatch::cat, ret::reference);
  m.def("reshape", &ir::dispatch::reshape, ret::reference);
  typedef std::tuple<ir::value *, ir::value *> (*broadcast_ty)(ir::value *, ir::value *, ir::builder *);
  typedef ir::value *(*broadcast_to_ty)(ir::value *, ir::type::block_shapes_t, ir::builder *);
  m.def("broadcast", (broadcast_ty)(&ir::dispatch::broadcast), ret::reference);
  m.def("broadcast_to", (broadcast_to_ty)(&ir::dispatch::broadcast), ret::reference);
  m.def("bitcast", &ir::dispatch::bitcast, ret::reference);
  m.def("cast", &ir::dispatch::cast, ret::reference);
  // memory
  m.def("load", &ir::dispatch::load, ret::reference);
  m.def("store", &ir::dispatch::store, ret::reference);
  m.def("atomic_cas", &ir::dispatch::atomic_cas, ret::reference);
  m.def("atomic_xchg", &ir::dispatch::atomic_xchg, ret::reference);
  m.def("atomic_add", &ir::dispatch::atomic_add, ret::reference);
  m.def("atomic_max", &ir::dispatch::atomic_max, ret::reference);
  m.def("atomic_min", &ir::dispatch::atomic_min, ret::reference);
  m.def("atomic_and", &ir::dispatch::atomic_and, ret::reference);
  m.def("atomic_or", &ir::dispatch::atomic_or, ret::reference);
  m.def("atomic_xor", &ir::dispatch::atomic_xor, ret::reference);
  // linear algebra
  m.def("dot", &ir::dispatch::dot, ret::reference);
  // indexing
  m.def("where", &ir::dispatch::where, ret::reference);
  // reduction
  m.def("min", &ir::dispatch::min, ret::reference);
  m.def("max", &ir::dispatch::max, ret::reference);
  m.def("sum", &ir::dispatch::sum, ret::reference);
  m.def("xor_sum", &ir::dispatch::xor_sum, ret::reference);
  // math
  m.def("umulhi", &ir::dispatch::umulhi, ret::reference);
  m.def("exp", &ir::dispatch::exp, ret::reference);
  m.def("log", &ir::dispatch::log, ret::reference);
  m.def("cos", &ir::dispatch::cos, ret::reference);
  m.def("sin", &ir::dispatch::sin, ret::reference);
  m.def("sqrt", &ir::dispatch::sqrt, ret::reference);
  // utilities
  m.def("clock", &ir::dispatch::clock, ret::reference);
  m.def("globaltimer", &ir::dispatch::globaltimer, ret::reference);
  // internal (debugging only)
  m.def("multiple_of", &ir::dispatch::multiple_of, ret::reference);
  m.def("max_contiguous", &ir::dispatch::max_contiguous, ret::reference);
  m.def("debug_barrier", &ir::dispatch::debug_barrier, ret::reference);
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::class_<ir::context>(m, "context")
      .def(py::init<>());

  auto value = py::class_<ir::value>(m, "value");
  value.def_property("name", &ir::value::get_name, &ir::value::set_name);
  value.def_property_readonly("type", &ir::value::get_type);

  py::class_<ir::user, ir::value>(m, "user");

  py::class_<ir::constant, ir::user>(m, "constant");

  py::class_<ir::undef_value, ir::constant>(m, "undef")
      .def("get", &ir::undef_value::get, ret::reference);

  py::class_<ir::constant_int, ir::constant>(m, "constant_int")
      .def_property_readonly("value", &ir::constant_int::get_value)
      .def("__int__", [](ir::constant_int *self) { return self->get_value(); })
      .def("__bool__", [](ir::constant_int *self) { return self->get_value(); });

  py::class_<ir::constant_fp, ir::constant>(m, "constant_float")
      .def_property_readonly("value", &ir::constant_fp::get_value);

  py::class_<ir::instruction, ir::user>(m, "instruction");
  py::class_<ir::phi_node, ir::user>(m, "phi_node");

  py::class_<ir::type>(m, "type")
      .def("is_ptr", &ir::type::is_pointer_ty)
      .def("is_int", static_cast<bool (ir::type::*)() const>(&ir::type::is_integer_ty))
      .def("is_floating", &ir::type::is_floating_point_ty)
      .def("is_block", &ir::type::is_block_ty)
      .def("make_ptr", &ir::pointer_type::get, ret::reference)
      .def("make_function", &ir::function_type::get, ret::reference)
      .def("make_block", &ir::block_type::get, ret::reference)
      .def("get_void", &ir::type::get_void_ty, ret::reference)
      .def("get_fp8", &ir::type::get_fp8_ty, ret::reference)
      .def("get_fp16", &ir::type::get_fp16_ty, ret::reference)
      .def("get_bf16", &ir::type::get_bf16_ty, ret::reference)
      .def("get_fp32", &ir::type::get_fp32_ty, ret::reference)
      .def("get_fp64", &ir::type::get_fp64_ty, ret::reference)
      .def("get_int1", &ir::type::get_int1_ty, ret::reference)
      .def("get_int8", &ir::type::get_int8_ty, ret::reference)
      .def("get_int16", &ir::type::get_int16_ty, ret::reference)
      .def("get_int32", &ir::type::get_int32_ty, ret::reference)
      .def("get_int64", &ir::type::get_int64_ty, ret::reference)
      .def("get_uint8", &ir::type::get_uint8_ty, ret::reference)
      .def("get_uint16", &ir::type::get_uint16_ty, ret::reference)
      .def("get_uint32", &ir::type::get_uint32_ty, ret::reference)
      .def("get_uint64", &ir::type::get_uint64_ty, ret::reference)

      .def("is_void", &ir::type::is_void_ty)
      .def("is_fp8", &ir::type::is_fp8_ty)
      .def("is_fp16", &ir::type::is_fp16_ty)
      .def("is_bf16", &ir::type::is_bf16_ty)
      .def("is_fp32", &ir::type::is_fp32_ty)
      .def("is_fp64", &ir::type::is_fp64_ty)
      .def("is_int1", [](ir::type *self) { return self->is_integer_ty(1, ir::signedness::SIGNED); })
      .def("is_int8", [](ir::type *self) { return self->is_integer_ty(8, ir::signedness::SIGNED); })
      .def("is_int16", [](ir::type *self) { return self->is_integer_ty(16, ir::signedness::SIGNED); })
      .def("is_int32", [](ir::type *self) { return self->is_integer_ty(32, ir::signedness::SIGNED); })
      .def("is_int64", [](ir::type *self) { return self->is_integer_ty(64, ir::signedness::SIGNED); })
      .def("is_uint8", [](ir::type *self) { return self->is_integer_ty(8, ir::signedness::UNSIGNED); })
      .def("is_uint16", [](ir::type *self) { return self->is_integer_ty(16, ir::signedness::UNSIGNED); })
      .def("is_uint32", [](ir::type *self) { return self->is_integer_ty(32, ir::signedness::UNSIGNED); })
      .def("is_uint64", [](ir::type *self) { return self->is_integer_ty(64, ir::signedness::UNSIGNED); })

      .def("repr", &ir::type::repr)
      .def_property_readonly("fp_mantissa_width", &ir::type::get_fp_mantissa_width)
      .def_property_readonly("scalar", &ir::type::get_scalar_ty)
      .def_property_readonly("context", &ir::type::get_context, ret::reference);

  py::class_<ir::pointer_type, ir::type>(m, "pointer_type")
      .def_property_readonly("element", &ir::pointer_type::get_element_ty, ret::reference);

  py::class_<ir::function_type, ir::type>(m, "function_type");
  py::class_<ir::integer_type, ir::type>(m, "integer_type");
  py::class_<ir::block_type, ir::type>(m, "block_type")
      .def_property_readonly("shape", &ir::block_type::get_shapes)
      .def_property_readonly("numel", &ir::type::get_tile_num_elements);

  py::class_<ir::module>(m, "module")
      .def(py::init<std::string, ir::builder &>())
      .def("get_or_insert_function", &ir::module::get_or_insert_function, ret::reference)
      .def("seal_block", &ir::module::seal_block)
      .def("set_value", (void (ir::module::*)(const std::string &, ir::value *)) & ir::module::set_value)
      .def("set_type", &ir::module::set_type)
      .def("get_value", (ir::value * (ir::module::*)(const std::string &)) & ir::module::get_value, ret::reference)
      .def("get_values", &ir::module::get_values, ret::reference)
      .def("set_values", &ir::module::set_values)
      .def("get_types", &ir::module::get_types, ret::reference)
      .def("set_types", &ir::module::set_types)
      .def_property_readonly("builder", &ir::module::get_builder, ret::reference);

  using eattr = ir::attribute_kind_t;
  py::enum_<eattr>(m, "attribute_kind")
      .value("readonly", eattr::readonly)
      .value("writeonly", eattr::writeonly)
      .value("noalias", eattr::noalias)
      .value("aligned", eattr::aligned)
      .value("multiple_of", eattr::multiple_of)
      .value("retune", eattr::retune)
      .value("not_implemented", eattr::not_implemented);

  py::class_<ir::attribute>(m, "attribute")
      .def(py::init<eattr, int>());

  py::class_<ir::function>(m, "function")
      .def_property_readonly("args", &ir::function::args)
      .def_property_readonly("attrs", &ir::function::attrs)
      .def("add_attr", &ir::function::add_attr);

  py::class_<ir::argument, ir::value>(m, "argument");

  py::class_<ir::basic_block, ir::value>(m, "basic_block")
      .def("create", &ir::basic_block::create, ret::reference)
      .def_property_readonly("parent", &ir::basic_block::get_parent, ret::reference);

  py::class_<ir::builder>(m, "builder", py::dynamic_attr())
      .def(py::init<ir::context &>())
      // getters
      .def_property_readonly("context", &ir::builder::get_context, ret::reference)
      // control flow
      .def("br", &ir::builder::create_br, ret::reference)
      .def("cond_br", &ir::builder::create_cond_br, ret::reference)
      .def("ret_void", &ir::builder::create_ret_void, ret::reference)
      .def("get_insert_block", &ir::builder::get_insert_block, ret::reference)
      .def("set_insert_block", (void (ir::builder::*)(ir::basic_block *)) & ir::builder::set_insert_point)
      // constants
      .def("get_int1", &ir::builder::get_int1, ret::reference)
      .def("get_int32", &ir::builder::get_int32, ret::reference)
      .def("get_int64", &ir::builder::get_int64, ret::reference)
      .def("get_uint32", &ir::builder::get_uint32, ret::reference)
      .def("get_uint64", &ir::builder::get_uint64, ret::reference)
      .def("get_float16", &ir::builder::get_float16, ret::reference)
      .def("get_float32", &ir::builder::get_float32, ret::reference)
      .def("get_range", &ir::builder::get_range, ret::reference);
}

void init_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  init_triton_codegen(std::move(subm.def_submodule("code_gen")));
  init_triton_runtime(std::move(subm.def_submodule("runtime")));
  init_triton_ir(std::move(subm.def_submodule("ir")));
  init_triton_frontend(std::move(subm.def_submodule("frontend")));
}
