#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
#include <regex>
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
namespace lng = triton::lang;

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
                 const rt::options_t& opt,
                 const rt::function::autotune_vals_t& autotune_vals,
                 const std::vector<std::string>& autotune_key) {
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
    id_fn_map[op_id].reset(new rt::function(src, opt, &*tt_devices[dev_id], autotune_vals, autotune_key));
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


std::string extract_kernels(const std::string& str, const std::vector<std::string>& names) {
  if(names.empty())
    return str;
  // search for all regex matches of kernel_regex in str
  std::smatch matches;
  std::regex regex(" *__global__ +void +([_a-zA-Z][_a-zA-Z0-9]{0,30})");
  std::sregex_iterator it(str.begin(), str.end(), regex);
  std::sregex_iterator end;
  std::vector<std::tuple<std::string, int, int>> kernels;
  for (; it != end; ++it) {
    int pos = it->position();
    int len = it->length();
    std::string name = it->str(1);
    kernels.push_back(std::make_tuple(name, pos, len));
  }

  for(const std::string& name: names) {
    // check that str matches any string in kernels using std::any_of
    auto pred = [&name](const std::tuple<std::string, int, int>& t) { return std::get<0>(t) == name; };
    bool found = std::any_of(kernels.begin(), kernels.end(), pred);
    if(!found) throw std::runtime_error("Unable to find kernel `" + name + "` in provided source code:\n" + str);
  }


  // extract functions
  std::string ret;
  for(const auto& k: kernels) {
    std::string name;
    int pos, len;
    std::tie(name, pos, len) = k;
    if(std::find(names.begin(), names.end(), name) != names.end()){
      std::string def = str.substr(pos, str.size() - pos);
      int count, pos;
      // skip over declaration
      count = 1;
      pos = def.find('(');
      while(!(def[pos++] == ')' && count == 0) && pos < def.size()){
        count += def[pos] == '(';
        count -= def[pos] == ')';
      }
      // skip over definition
      count = 1;
      pos = def.find('{', pos);
      while(!(def[pos++] == '}' && count == 0) && pos < def.size()){
        count += def[pos] == '{';
        count -= def[pos] == '}';
      }
      ret += def.substr(0, pos);
      ret += '\n';
    }
  }

  return ret;
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
        .def(pybind11::init<>())
        .def_readwrite("defines"  , &rt::options_t::defines)
        .def_readwrite("num_warps", &rt::options_t::num_warps);

    // hooks into triton constructs since frameworks may not use pybind11
    m.def("extract_kernels", &extract_kernels);
    m.def("get_fn_signature", &get_fn_signature);
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
