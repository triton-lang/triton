#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <regex>
#include <algorithm>
#include "triton/codegen/selection/selection.h"
#include "triton/runtime/function.h"
#include "triton/lang/lang.h"
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/tools/bench.hpp"

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern triton::lang::translation_unit *ast_root;

using namespace triton;

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
    return "float32";
  if(ty->is_double_ty())
    return "float64";
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

inline triton::lang::translation_unit *make_ast(const char *src) {
  YY_BUFFER_STATE buffer = yy_scan_string(src);
  yyparse();
  yy_delete_buffer(buffer);
  triton::lang::translation_unit *program = ast_root;
  return program;
}

inline std::unique_ptr<ir::module> make_ir(ir::context& ctx, triton::lang::translation_unit *program) {
  // create Triton-IR from AST
  ir::module* module = new ir::module("", ctx);
  program->codegen(module);
  return std::unique_ptr<ir::module>(module);
}


void gen_extract_inputs(std::ostream &os, const std::vector<ir::argument*>& args) {
  for(unsigned i = 0; i < args.size(); i++){
    ir::value *arg = args[i];
    std::string suffix = "";
    ir::type *tr_ty = arg->get_type();
    std::string tf_ty = ref_to_tf_ty(tr_ty);
    if(!tr_ty->is_pointer_ty())
      suffix = ".scalar<" + tf_ty + ">()()";
    os << "  " << tf_ty << " " << arg->get_name() << " = context->input(" << i << ")" << suffix << ";\n  ";
  }
}

void gen_set_outputs(std::ostream &os, const std::vector<std::string>& outputs) {
  for(unsigned i = 0; i < outputs.size(); i++)
    os << "  context->set_output(" << i << ", " << outputs[i] << ");\n  ";
}

void gen_make_handles(std::ostream &os, const std::vector<ir::argument*>& args) {
  for(unsigned i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    if(!arg->get_type()->is_pointer_ty())
      continue;
    const std::string& name = arg->get_name();
    os << "  drv::cu_buffer cu_" + name + "(ctx, " + name + ".tensor_data().size(), (CUdeviceptr)" + name + ".tensor_data().data(), false);\n  ";
  }
}

void gen_make_spmd_grid(std::ostream &os, const std::vector<std::string>& macros) {
  std::regex regex("#([a-zA-Z]([a-zA-Z]|[0-9])*)");
  std::vector<std::string> grids = macros;
  for(size_t i = grids.size(); i < 3; i++)
    grids.push_back("1");
  std::string grid = "rt::grid_t{";
  for(size_t i = 0; i < grids.size(); i++){
    if(i > 0)
      grid += ", ";
    grid += std::regex_replace(grids[i], regex, "x.at(\"$1\")");
  }
  grid += "}";

  os << "  auto grid = [&](const rt::params_t& x) { return " << grid << "; };\n  ";
}

void gen_make_launch_function(std::ostream &os, const std::vector<ir::argument*>& args) {
  os << "  fn_({";
  for(unsigned i = 0; i < args.size() ; i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    if(arg->get_type()->is_pointer_ty())
      name = "&cu_" + name;
    if(i > 0)
      os << ", ";
    os << name;
  }
  os << "}, grid, stream);  \n";
}

void gen_register_kernel_builder(std::ostream &os, const std::string &name,
                                 const std::string &opname,
                                 const std::vector<ir::argument*>& args){
  os << "REGISTER_KERNEL_BUILDER(Name(\"" + name + "\").Device(DEVICE_GPU)";
  for(size_t i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    auto tolower = [](char c) { return std::tolower(c);};
    std::transform(name.begin(), name.end(), name.begin(), tolower);
    if(!arg->get_type()->is_pointer_ty())
      os << ".HostMemory(\"" + name + "\")";
  }
  os <<  ", " + opname << ");\n";
}

void gen_register_op(std::ostream &os, const std::string &name,
                     const std::vector<ir::argument*>& args,
                     const std::vector<std::string>& outputs){
  os << "REGISTER_OP(\"" << name << "\")\n";
  for(size_t i = 0; i < args.size(); i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    auto tolower = [](char c) { return std::tolower(c);};
    std::transform(name.begin(), name.end(), name.begin(), tolower);
    os << "  .Input(\"" << name << ": " << to_tf_scalar_ty(arg->get_type()) << "\")\n";
  }
  for(size_t i = 0; i < outputs.size(); i++){
    std::string name = outputs[i];
    size_t idx;
    for(idx = 0; idx < args.size(); idx++)
      if(args[idx]->get_name() == name)
        break;
    if(idx == args.size())
      throw std::runtime_error("unknown output");
    os << "  .Output(\"out" << i << ": " << to_tf_scalar_ty(args[idx]->get_type()) << "\")\n";
  }
  os << ";\n";
}

std::string make_tensorflow_src(const std::string src,
                                const std::vector<std::string>& outputs,
                                const std::vector<std::string>& macros) {
  triton::lang::translation_unit *ast = make_ast(src.c_str());
  triton::ir::context context;
  std::unique_ptr<ir::module> ir = make_ir(context, ast);
  // function
  ir::function* fn = ir->get_function_list().front();
  std::string name = fn->get_name();
  name[0] = static_cast<char>(std::toupper(name[0]));
  std::string opname = name + "Op";

  std::ostringstream oss;
  oss << R"(
#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
namespace rt = triton::runtime;
namespace drv = triton::driver;

std::string src = R"TTKERNSRC( )" + src + ")TTKERNSRC\";" + R"(

class )" << opname << R"(: public OpKernel {
 public:
  explicit )" << opname << R"((OpKernelConstruction* context)
    : OpKernel(context), fn_(src) { }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    drv::cu_stream sstream(device.stream(), false);
    drv::context* ctx = sstream.context();
    drv::stream* stream = &sstream;
    // extract inputs
    )";
gen_extract_inputs(oss, fn->args());
oss << R"(
    // set outputs
    )";
gen_set_outputs(oss, outputs);
oss << R"(
    // wrap tensors
    )";
gen_make_handles(oss, fn->args());
oss << R"(
    // create spmd grid
    )";
gen_make_spmd_grid(oss, macros);
oss << R"(
    // launch function
    )";
gen_make_launch_function(oss, fn->args());
oss << R"(
  }

private:
  rt::function fn_;
};

// register kernel builder
)";
gen_register_kernel_builder(oss, name, opname, fn->args());
oss << R"(
// register op
)";
gen_register_op(oss, name, fn->args(), outputs);

  return oss.str();
}


PYBIND11_MODULE(libtriton, m) {
    m.doc() = "Python bindings to the C++ Triton API";
    m.def("make_tensorflow_src", &make_tensorflow_src, "Creates C++ source code for a custom Tensorflow op corresponding to the specified Triton kernel");
}
