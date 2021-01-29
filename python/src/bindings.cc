#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include "triton/runtime/arg.h"
#include "triton/lang/code_gen.h"
#include "triton/lang/parser.h"
#include "triton/lang/cpp.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"

using namespace triton;
namespace rt = triton::runtime;
namespace drv = triton::driver;

typedef std::pair<int, int> map_key_t;

std::map<map_key_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
std::map<int, std::shared_ptr<rt::function>> id_fn_map;
std::map<int, std::shared_ptr<triton::driver::device>> tt_devices;
std::map<int, std::shared_ptr<triton::driver::stream>> tt_streams;
std::unordered_map<const rt::options_t*, pybind11::object> opt_cache_;
extern CUstream torch_get_cuda_stream(int64_t dev_id);
extern CUdevice torch_get_cuda_device(int64_t dev_id);


/* Grid utilities */

void register_grid(const map_key_t& key,
                   const rt::function::grid_fn_ty& grid_fn) {
  id_grid_map[key].reset(new rt::function::grid_fn_ty(grid_fn));
}

void delete_grid(const map_key_t& key) {
  id_grid_map.erase(key);
}

/* Function utilities */

void register_fn(int op_id, 
                 int dev_id,
                 const std::string& src,
                 const rt::options_space_t& opt) {
  if(tt_devices.find(dev_id) == tt_devices.end()) {
    driver::device* device;
    driver::stream* stream;
    if(dev_id >= 0){
      device = new triton::driver::cu_device(torch_get_cuda_device(dev_id), false);
      stream = new triton::driver::cu_stream(torch_get_cuda_stream(dev_id), false);
    }
    else{
      device = new triton::driver::host_device();
      stream = new triton::driver::host_stream();
    }
    tt_devices[dev_id].reset(device);
    tt_streams[dev_id].reset(stream);
  }
  if(id_fn_map.find(op_id) == id_fn_map.end()){
    id_fn_map[op_id].reset(new rt::function(src, opt, &*tt_devices[dev_id]));
  }
  for(const auto& k: id_fn_map[op_id]->get_kernels()){
    const rt::options_t* opt = &k.first;
    pybind11::object obj = pybind11::cast(opt, pybind11::return_value_policy::reference);
    for(auto x: opt->defines)
     if(std::all_of(x.second.begin(), x.second.end(), ::isdigit))
       obj.attr(x.first.c_str()) = std::stoi(x.second);
    opt_cache_[&k.second->opt] = obj;
  }

}

void delete_fn(int op_id) {
  id_fn_map.erase(op_id);
}


void cleanup() {
  id_grid_map.clear();
  id_fn_map.clear();
  opt_cache_.clear();
}

size_t make_op_id() {
  return id_fn_map.size();
}

std::vector<rt::arg_type> get_fn_signature(size_t op_id) {
  return id_fn_map[op_id]->get_kernels()[0].second->get_sig();
}

void launch_kernel(int64_t op_id, int64_t dev_id, const std::string& args, size_t grid_0, size_t grid_1, size_t grid_2){
  rt::function* fn = id_fn_map.at(op_id).get();
  (*fn)((void**)args.c_str(), args.size(), {grid_0, grid_1, grid_2}, &*tt_streams[dev_id]);

  // for(size_t n = 0; n < constant_names.size(); n++){
  // const torch::Tensor& x = constant_vals[n];
  // fn->set_cst(constant_names[n].c_str(), (char*)x.data_ptr(), x.numel()*x.element_size());
}

pybind11::object autotune(int64_t op_id, int64_t dev_id, const std::string& args, const rt::function::grid_fn_ty& grid){
  rt::function* fn = id_fn_map.at(op_id).get();
  auto wrapper = [&grid](const rt::options_t& opt){
    pybind11::object obj = pybind11::cast(&opt, pybind11::return_value_policy::reference);
    for(auto x: opt.defines)
     if(std::all_of(x.second.begin(), x.second.end(), ::isdigit))
       obj.attr(x.first.c_str()) = std::stoi(x.second);
    return grid(*obj.cast<rt::options_t*>());
  };
  rt::kernel* kernel = fn->autotune((void**)args.c_str(), args.size(), wrapper, &*tt_streams[dev_id]);
  return opt_cache_.at(&kernel->opt);
}


void init_superblocking(pybind11::module &m);
void init_launch(pybind11::module &m);

PYBIND11_MODULE(libtriton, m) {
    m.doc() = "Python bindings to the C++ Triton API";

    // bindings for triton classes
    pybind11::enum_<rt::arg_type>(m, "arg_type")
        .value("int1"  , rt::INT1_T)
        .value("int8"  , rt::INT8_T)
        .value("int16" , rt::INT16_T)
        .value("int32" , rt::INT32_T)
        .value("int64" , rt::INT64_T)
        .value("half"  , rt::HALF_T)
        .value("float" , rt::FLOAT_T)
        .value("double", rt::DOUBLE_T)
        .value("buffer", rt::BUFFER_T);
    
    pybind11::enum_<rt::asm_mode_t>(m, "asm_mode")
        .value("ptx" , rt::ASM_NV_PTX)
        .value("sass", rt::ASM_NV_SASS);

    pybind11::class_<rt::options_t>(m, "options", pybind11::dynamic_attr())
        .def_readwrite("num_warps", &rt::options_t::num_warps)
        .def_readwrite("defines"  , &rt::options_t::defines);

    pybind11::class_<rt::options_space_t>(m, "options_space")
        .def(pybind11::init<>())
        .def_readwrite("num_warps", &rt::options_space_t::num_warps)
        .def_readwrite("defines"  , &rt::options_space_t::defines);

    // hooks into triton constructs since frameworks may not use pybind11
    m.def("get_fn_signature", &get_fn_signature);
    // m.def("get_fn_asm", &get_fn_asm);
    m.def("register_grid", &register_grid);
    m.def("delete_grid", &delete_grid);
    m.def("register_fn", &register_fn);
    m.def("delete_fn", &delete_fn);
    m.def("make_op_id", &make_op_id);
    m.def("cleanup", &cleanup);
    m.def("autotune", &autotune, pybind11::return_value_policy::reference);
    m.def("launch_kernel", &launch_kernel);

    init_launch(m);
    init_superblocking(m);
}
