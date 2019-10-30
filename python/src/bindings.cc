#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
#include <regex>
#include <algorithm>
#include "triton/runtime/function.h"
#include "triton/lang/code_gen.h"
#include "triton/lang/parser.h"
#include "triton/lang/cpp.h"
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/tools/bench.hpp"

using namespace triton;

namespace rt = triton::runtime;

std::map<size_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
std::map<size_t, std::shared_ptr<rt::function>> id_fn_map;
std::map<size_t, double> fp64scalar_map;
std::map<size_t, int64_t> i64scalar_map;

/* Grid map */

void register_grid(size_t id,
                   const rt::function::grid_fn_ty& grid_fn) {
  id_grid_map[id].reset(new rt::function::grid_fn_ty(grid_fn));
}

void delete_grid(size_t id) {
  id_grid_map.erase(id);
}

/* Function map */

void register_fn(size_t id,
                   const std::string& src,
                   const rt::function::options_space_t& opt) {
  id_fn_map[id].reset(new rt::function(src, opt));
}

void delete_fn(size_t id) {
  id_fn_map.erase(id);
}

void cleanup() {
  id_grid_map.clear();
  id_fn_map.clear();
  i64scalar_map.clear();
}

size_t make_op_id() {
  return id_fn_map.size();
}

/* TF scalar wrapper */
size_t make_scalar_id() {
  size_t ret = i64scalar_map.size();
  i64scalar_map[ret] = int64_t();
  return ret;
}

bool has_scalar(size_t id) {
  return i64scalar_map.find(id) != i64scalar_map.end();
}

int64_t retrieve_scalar(size_t id) {
  return i64scalar_map.at(id);
}

/* TF source-code generation */

inline std::string to_tf_ty(ir::type *ty) {
  if(ty->is_integer_ty(1))
    return "bool";
  if(ty->is_integer_ty(8))
    return "int8";
  if(ty->is_integer_ty(16))
    return "int16";
  if(ty->is_integer_ty(32))
    return "int32";
  if(ty->is_integer_ty(64))
    return "int64";
  if(ty->is_half_ty())
    return "float16";
  if(ty->is_float_ty())
    return "float";
  if(ty->is_double_ty())
    return "double";
  if(ty->is_pointer_ty())
    return "Tensor";
  throw std::runtime_error("unknown type");
}

inline std::string to_tf_scalar_ty(ir::type *ty) {
  if(ty->is_pointer_ty())
    return to_tf_ty(ty->get_pointer_element_ty());
  else {
    return to_tf_ty(ty);
  }
}

inline std::string ref_to_tf_ty(ir::type *ty) {
  std::string res = to_tf_ty(ty);
  if(ty->is_pointer_ty())
    res = "const " + res + "&";
  return res;
}


void gen_extract_inputs(std::ostream &os, const std::vector<ir::argument*>& args, const std::vector<std::string>& outputs) {
  for(unsigned i = 0; i < args.size(); i++){
    ir::value *arg = args[i];
    const std::string& name = arg->get_name();
    std::string ty = to_tf_ty(arg->get_type());
    if(!arg->get_type()->is_pointer_ty())
      os << "  " << ty << " " << name << " = context->input(" << i << ").scalar<" << ty << ">()();\n  ";
    else if(std::find(outputs.begin(), outputs.end(), arg->get_name()) == outputs.end())
      os << "  const Tensor* " << name << " = &context->input(" << i << ");\n  ";
    else
      os << "  Tensor* " << name << " = nullptr;\n  ";
  }
}

void gen_set_outputs(std::ostream &os, const std::vector<ir::argument*>& args, const std::vector<std::string>& outputs) {
  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  TensorShape shape" << i << ";\n  ";
  // initialize shapes

  std::vector<int> out_idx;
  for(size_t i = 0; i < outputs.size(); i++){
    std::string name = outputs[i];
    size_t idx;
    for(idx = 0; idx < args.size(); idx++)
      if(args[idx]->get_name() == name)
        break;
    if(idx == args.size())
      throw std::runtime_error("unknown output");
    out_idx.push_back(idx);
  }

  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  const Tensor& " << outputs[i] << "_shape = context->input(" << out_idx[i] << ");\n  ";
  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  const int32* " << outputs[i] << "_shape_data = (const int32*)" << outputs[i] << "_shape.tensor_data().data();\n  ";
  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  size_t " << outputs[i] << "_rank = " << outputs[i] << "_shape.dim_size(0);\n  ";
  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  for(size_t d = 0; d < " << outputs[i] << "_rank ; d++) "
       << "shape" << i << ".AddDim(" << outputs[i] << "_shape_data[d]);\n  ";
  
  // allocate
  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  OP_REQUIRES_OK(context, context->allocate_output(" << i << ", shape" << i << ", &" << outputs[i] << "));\n  ";
}

void gen_make_handles(std::ostream &os, const std::vector<ir::argument*>& args) {
  for(unsigned i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    if(!arg->get_type()->is_pointer_ty())
      continue;
    const std::string& name = arg->get_name();
    os << "  drv::cu_buffer cu_" + name + "(ctx, " + name + "->tensor_data().size(), (CUdeviceptr)" + name + "->tensor_data().data(), false);\n  ";
  }
}

void gen_make_launch_function(std::ostream &os, int num_outputs, const std::vector<ir::argument*>& args) {
  os << "  std::function<void()> run = [&](){\n  ";
  os << "    (*id_fn_map.at(id_))({";
  for(unsigned i = 0; i < args.size() ; i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    if(arg->get_type()->is_pointer_ty())
      name = "&cu_" + name;
    if(i > 0)
      os << ", ";
    os << name;
  }
  os << "}, *id_grid_map.at(id_), stream);\n";
  os << "  };\n  ";
  os << "  run();";
  os << "  if(bench_ > 0)\n  ";
  os << "    i64scalar_map[bench_id_] = triton::tools::bench(run, stream);\n  ";
}

void gen_tf_register_kernel_builder(std::ostream &os, const std::string &name,
                                 const std::string &opname,
                                 const std::vector<ir::argument*>& args,
                                 const std::vector<std::string>& outputs){

  auto tolower = [](char c) { return std::tolower(c);};
  os << "REGISTER_KERNEL_BUILDER(Name(\"" + name + "\").Device(DEVICE_GPU)";
  for(size_t i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    std::transform(name.begin(), name.end(), name.begin(), tolower);
    if(!arg->get_type()->is_pointer_ty())
      os << ".HostMemory(\"" + name + "\")";
  }
  for(size_t i = 0; i < outputs.size(); i++){
    std::string name = outputs[i];
    std::transform(name.begin(), name.end(), name.begin(), tolower);
    os << ".HostMemory(\"" << name << "_shape\")";
  }
  os <<  ", " + opname << ");\n";
}

void gen_tf_register_op(std::ostream &os, const std::string &name,
                     const std::vector<ir::argument*>& args,
                     const std::vector<std::string>& outputs){
  
  auto tolower = [](char c) { return std::tolower(c);};
  
  os << "REGISTER_OP(\"" << name << "\")\n";
  for(size_t i = 0; i < args.size(); i++)
    os << "  .Attr(\"T" << i << " : {bool, int8, int16, int32, int64, float16, float32, float64}\")" << std::endl;
  for(size_t i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    std::transform(name.begin(), name.end(), name.begin(), tolower);
    if(std::find(outputs.begin(), outputs.end(), arg->get_name()) == outputs.end())
      os << "  .Input(\"" << name << ": T" << i << "\")\n";
    else
      os << "  .Input(\"" << name << "_shape: int32\")\n";
  }
  std::vector<int> out_idx;
  for(size_t i = 0; i < outputs.size(); i++){
    std::string name = outputs[i];
    size_t idx;
    for(idx = 0; idx < args.size(); idx++)
      if(args[idx]->get_name() == name)
        break;
    if(idx == args.size())
      throw std::runtime_error("unknown output");
    out_idx.push_back(idx);
  }
  for(size_t i = 0; i < out_idx.size(); i++){
    std::string name = outputs[i];
    std::transform(name.begin(), name.end(), name.begin(), tolower);
    os << "  .Output(\"" << name << ": T" << out_idx[i] << "\")\n";
  }
  os << "  .Attr(\"id: int\")\n";
  os << "  .Attr(\"bench: int\")\n";
  os << "  .Attr(\"bench_id: int\")\n";
  os << "  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* ctx) {\n";
  for(size_t i = 0; i < out_idx.size(); i++)
    os << "    shape_inference::ShapeHandle handle" << i << ";\n";
  for(size_t i = 0; i < out_idx.size(); i++)
    os << "    ctx->MakeShapeFromShapeTensor(" << out_idx[i] << ", &handle" << i << ");\n";
  for(size_t i = 0; i < out_idx.size(); i++)
    os << "    ctx->set_output(" << i << ", handle" << i << ");\n";
  os << "      return Status::OK();\n";
  os << "  })\n";

  os << ";\n";
}

void make_module(const std::string& src, ir::module* ir,
                 const runtime::function::options_space_t& opt) {
  std::string copy = triton::runtime::function::preheader() + src;
  // pre-process
  TokenSequence tokens;
  Preprocessor cpp(&copy, true);
  for(auto it: opt.defines){
    cpp.AddMacro(it.first, &it.second[0]);
  }
  cpp.Process(tokens);
  // parse
  Parser parser(tokens);
  parser.Parse();
  Generator gen(&parser);
  gen.Gen(ir);
}

std::tuple<std::string,
           std::string> make_tensorflow_src(const std::string& src,
                                const std::vector<std::string>& outputs,
                                const runtime::function::options_space_t& opt)
{
  // triton-ir code-gen
  ir::context ctx;
  auto ir = std::shared_ptr<ir::module>(new ir::module("", ctx));
  make_module(src, &*ir, opt);

  // function
  ir::function* fn = ir->get_function_list().front();
  std::string name = fn->get_name();
  std::string cc_name = name;
  cc_name[0] = static_cast<char>(std::toupper(cc_name[0]));
  std::string opname = cc_name + "Op";

  std::ostringstream oss;
  oss << R"(
#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include "triton/tools/bench.hpp"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
namespace rt = triton::runtime;
namespace drv = triton::driver;

extern std::map<size_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
extern std::map<size_t, std::shared_ptr<rt::function>> id_fn_map;
extern std::map<size_t, int64_t> i64scalar_map;

class )" << opname << R"(: public OpKernel {
 public:
  explicit )" << opname << R"((OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    OP_REQUIRES_OK(context, context->GetAttr("bench", &bench_));
    OP_REQUIRES_OK(context, context->GetAttr("bench_id", &bench_id_));
  }

  void Compute(OpKernelContext* context){

    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    drv::cu_stream sstream(device.stream(), false);
    drv::context* ctx = sstream.context();
    drv::stream* stream = &sstream;
    
    // extract inputs
    )";
gen_extract_inputs(oss, fn->args(), outputs);
oss << R"(
    // set outputs
    )";
gen_set_outputs(oss, fn->args(), outputs);
oss << R"(
    // wrap tensors
    )";
gen_make_handles(oss, fn->args());
oss << R"(
    )";
oss << R"(
    // launch function
    )";
gen_make_launch_function(oss, outputs.size(), fn->args());
oss << R"(
  }

private:
  int id_;
  int bench_;
  int64 bench_id_;
};

// register kernel builder
)";
gen_tf_register_kernel_builder(oss, cc_name, opname, fn->args(), outputs);
oss << R"(
// register op
)";
gen_tf_register_op(oss, cc_name, fn->args(), outputs);

  return {oss.str(), name};
}


inline std::string to_torch_ty(ir::type *ty) {
  if(ty->is_integer_ty())
    return "int64_t";
  if(ty->is_half_ty())
    return "double";
  if(ty->is_float_ty())
    return "double";
  if(ty->is_double_ty())
    return "double";
  if(ty->is_pointer_ty())
    return "torch::Tensor";
  throw std::runtime_error("unknown type");
}

inline std::string to_c_ty(ir::type *ty) {
  if(ty->is_integer_ty(1))
    return "bool";
  if(ty->is_integer_ty(8))
    return "int8_t";
  if(ty->is_integer_ty(16))
    return "int16_t";
  if(ty->is_integer_ty(32))
    return "int32_t";
  if(ty->is_integer_ty(64))
    return "int64_t";
  if(ty->is_half_ty())
    return "half";
  if(ty->is_float_ty())
    return "float";
  if(ty->is_double_ty())
    return "double";
  if(ty->is_pointer_ty())
    return "drv::cu_buffer";
  throw std::runtime_error("unknown type");
}



void gen_torch_signature(std::ostringstream& oss,
                         ir::function* fn,
                         const std::vector<std::string>& outputs,
                         const std::string& name) {
  const auto& args = fn->args();
  std::vector<ir::type*> out_types;
  for(const std::string& out: outputs) {
    auto it = std::find_if(args.begin(), args.end(),
                           [&](ir::argument* arg) { return arg->get_name() == out; });
    if(it == args.end())
      throw std::runtime_error("unknown argument");
    out_types.push_back((*it)->get_type());
  }

  std::string ret_ty;
  if(out_types.empty())
    ret_ty = "void";
  else{
    ir::type* ty = out_types[0];
    ret_ty = to_torch_ty(ty);
    if(out_types.size() > 1){
      for(size_t i = 1; i < out_types.size(); i++)
        if(out_types[i] != ty)
          throw std::runtime_error("outputs of different types not supported by pytorch");
      ret_ty = "std::vector<" + ret_ty + ">";
    }
  }

  oss << ret_ty << " " << name << "(";
  oss << "int64_t id, ";
  oss << "int64_t bench, ";
  oss << "int64_t bench_id,  ";
  for(size_t i = 0; i < args.size(); i++) {
    ir::argument* arg = args[i];
    if(i > 0)
      oss << ", ";
    oss << to_torch_ty(arg->get_type()) << " " << arg->get_name();
  }
  oss << ")";
}

void gen_torch_init_driver(std::ostringstream &oss,
                           const std::vector<ir::argument*>&args) {
  ir::argument* tensor = nullptr;
  for(ir::argument* arg: args)
    if(arg->get_type()->is_pointer_ty()){
      tensor = arg;
      break;
    }
  oss <<  "  // Wrap CUDA handles" << std::endl;
  oss <<  "  c10::DeviceIndex device = " << tensor->get_name() << ".storage().device().index();" << std::endl;
  oss <<  "  // Get stream" << std::endl;
  oss <<  "  CUstream custream = (CUstream)at::cuda::getCurrentCUDAStream(device).stream();" << std::endl;
  oss <<  "  triton::driver::cu_stream stream(custream, false);" << std::endl;
  oss <<  "  triton::driver::context* ctx = stream.context();" << std::endl;
}

void gen_torch_make_handles(std::ostream &os,
                            const std::vector<ir::argument*>& args) {
  for(unsigned i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    const std::string& name = arg->get_name();
    ir::type* ty = arg->get_type();
    if(!ty->is_pointer_ty())
      os << "  " << to_c_ty(ty) << " arg_" << name << " = " << name << ";" << std::endl;
    else{
      os << "  CHECK_INPUT(" << name << ");" << std::endl;
      os << "  drv::cu_buffer arg_" + name + "(ctx, " + name + ".storage().size(), (CUdeviceptr)" + name + ".storage().data(), false);" << std::endl;
    }
  }
}

void gen_torch_make_launch_function(std::ostream &os, const std::vector<ir::argument*>& args) {
  os << "  std::function<void()> run = [&](){\n  ";
  os << "    (*id_fn_map.at(id))({";
  for(unsigned i = 0; i < args.size() ; i++){
    ir::argument *arg = args[i];
    std::string name = "arg_" + arg->get_name();
    if(arg->get_type()->is_pointer_ty())
      name = "&" + name;
    if(i > 0)
      os << ", ";
    os << name;
  }
  os << "}, *id_grid_map.at(id), &stream);\n";
  os << "  };\n  ";
  os << "  run();";
  os << "  if(bench > 0)\n  ";
  os << "    i64scalar_map[bench_id] = triton::tools::bench(run, &stream);\n  ";
  }

void gen_torch_ret(std::ostream &os, const std::vector<std::string>& outputs) {
  if(outputs.size() == 1){
    os << "  return " << outputs[0] << ";" << std::endl;
    return;
  }
  os << "  return {";
  for(size_t i = 0; i < outputs.size(); i++){
    if(i > 0)
      os << ", ";
    os << outputs[i];
  }
  os << "};" << std::endl;
}

std::tuple<std::string,
           std::string> make_torch_src(const std::string& src,
                                         const std::vector<std::string>& outputs,
                                         const runtime::function::options_space_t& opt) {
  // triton-ir code-gen
  ir::context ctx;
  auto ir = std::shared_ptr<ir::module>(new ir::module("", ctx));
  make_module(src, &*ir, opt);
  // function
  ir::function* fn = ir->get_function_list().front();
  std::string name = fn->get_name();
  // generate framework code
  std::ostringstream oss;
  oss << R"(
#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include "triton/tools/bench.hpp"
#include "torch/extension.h"
#include "torch/script.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/CUDAHooks.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace rt = triton::runtime;
namespace drv = triton::driver;

extern std::map<size_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
extern std::map<size_t, std::shared_ptr<rt::function>> id_fn_map;
extern std::map<size_t, int64_t> i64scalar_map;

)";

  gen_torch_signature(oss, fn, outputs, name);
  oss << " {" << std::endl;
  gen_torch_init_driver(oss, fn->args());
  gen_torch_make_handles(oss, fn->args());
  gen_torch_make_launch_function(oss, fn->args());
  gen_torch_ret(oss, outputs);
  oss << "}" << std::endl;

  oss << std::endl;
  oss << std::endl;
  oss << "static auto registry = torch::RegisterOperators(\"triton::" << name << "\", &" << name << ");" << std::endl;

  return {oss.str(), name};
}


typedef triton::runtime::function::options_t options_t;
typedef triton::runtime::function::options_space_t options_space_t;

PYBIND11_MODULE(libtriton, m) {
    m.doc() = "Python bindings to the C++ Triton API";

    // framework binding source code generation
    m.def("make_tensorflow_src", &make_tensorflow_src,
          "Creates C++ source code for a custom Tensorflow op "
          "corresponding to the specified Triton kernel");
    m.def("make_torch_src", &make_torch_src,
          "Creates C++ source code for a custom PyTorch op ");

    // bindings for triton classes
    pybind11::class_<options_t>(m, "options")
        .def(pybind11::init<>())
        .def("d", &options_t::D<int>)
        .def_readonly("num_warps", &options_t::num_warps);

    pybind11::class_<options_space_t>(m, "options_space")
        .def(pybind11::init<>())
        .def_readwrite("defines", &options_space_t::defines)
        .def_readwrite("num_warps", &options_space_t::num_warps);

    // hooks into triton constructs since frameworks may not use pybind11
    m.def("register_grid", &register_grid);
    m.def("delete_grid", &delete_grid);
    m.def("register_fn", &register_fn);
    m.def("delete_fn", &delete_fn);
    m.def("make_op_id", &make_op_id);
    m.def("make_scalar_id", &make_scalar_id);
    m.def("retrieve_scalar", &retrieve_scalar);
    m.def("cleanup", &cleanup);
    ;
}
