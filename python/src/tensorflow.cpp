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

std::string make_tensorflow_src(const std::string src,
                                const std::vector<std::string>& outputs,
                                const std::vector<std::string>& macros) {
  triton::lang::translation_unit *ast = make_ast(src.c_str());
  triton::ir::context context;
  std::unique_ptr<ir::module> ir = make_ir(context, ast);
  // extract function signature
  ir::function* fn = ir->get_function_list().front();
  ir::function_type* fn_ty = fn->get_fn_type();
  // numberof arguments
  size_t n_args = fn_ty->get_num_params();
  size_t n_outputs = outputs.size();
  // extract function name
  std::string name = fn->get_name();
  name[0] = static_cast<char>(std::toupper(name[0]));
  std::string classname = name + "Op";
  // extract argument name
  std::vector<std::string> arg_names;
  for(ir::argument *arg: fn->args())
    arg_names.push_back(arg->get_name());
  // cached int to str
  std::vector<std::string> str_i;
  for(size_t i = 0; i < fn_ty->get_num_params(); i++)
    str_i.push_back(std::to_string(i));
  // index of tensors
  std::vector<size_t> ptr_idx;
  for(unsigned i = 0; i < fn_ty->get_num_params(); i++)
    if(fn_ty->get_param_ty(i)->is_pointer_ty())
      ptr_idx.push_back(i);
  // extract tensorflow types
  std::vector<std::string> tf_scalar_tys;
  std::transform(fn_ty->params_begin(), fn_ty->params_end(), std::back_inserter(tf_scalar_tys), to_tf_scalar_ty);
  std::vector<std::string> tf_cref_tys;
  std::transform(fn_ty->params_begin(), fn_ty->params_end(), std::back_inserter(tf_cref_tys), ref_to_tf_ty);
  // output indices
  std::vector<long> out_idx;
  for(const std::string &name : outputs){
    auto it = std::find(arg_names.begin(), arg_names.end(), name);
    out_idx.push_back(std::distance(arg_names.begin(), it));
  }
  std::ostringstream oss;

  std::string result = R"(
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

class )" + classname + R"(: public OpKernel {
 public:
  explicit )" + classname + R"((OpKernelConstruction* context)
    : OpKernel(context), fn_(src) { }

  void Compute(OpKernelContext* context){

    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    drv::cu_stream sstream(device.stream(), false);
    drv::context* ctx = sstream.context();
    drv::stream* stream = &sstream;

    // extract inputs)";
for(unsigned i = 0; i < n_args; i++){
  std::string suffix = "";
  std::string ty = tf_cref_tys[i];
  if(!fn_ty->get_param_ty(i)->is_pointer_ty())
    suffix = ".scalar<" + ty + ">()()";
  result += R"(
    )" + ty + " " + arg_names[i] + " = context->input(" + str_i[i] + ")" + suffix + ";";
}

result += R"(

    // extract outputs)";
for(unsigned i = 0; i < n_outputs; i++)
  result += R"(
   context->set_output()" + str_i[i] + ", " + outputs[i] + ");";

result += R"(

    // wrap tensors)";
for(size_t i: ptr_idx)
result += R"(
    drv::cu_buffer cu_)" + arg_names[i] + "(ctx, " + arg_names[i] + ".tensor_data().size(), (CUdeviceptr)" + arg_names[i] + R"(.tensor_data().data(), false);)";


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

result += R"(

    // create launch grid;
    auto grid = [&](const rt::params_t& x) { return )" + grid + R"(; };)";

result += R"(

    // execute function
    fn_({
    )";
for(unsigned i = 0; i < n_args; i++){
  std::string arg = arg_names[i];
  if(fn_ty->get_param_ty(i)->is_pointer_ty())
    arg = "&cu_" + arg;
  if(i > 0)
    result += ", ";
  result += arg;
}
result += R"(
    }, grid, stream);

  }

private:
  rt::function fn_;
};

REGISTER_KERNEL_BUILDER(Name(")" + name + "\").Device(DEVICE_GPU)";
for(size_t i = 0; i < tf_scalar_tys.size(); i++){
   std::string arg_name = arg_names[i];
  std::transform(arg_name.begin(), arg_name.end(), arg_name.begin(), [](char c) { return std::tolower(c);});
  if(!fn_ty->get_param_ty(i)->is_pointer_ty())
    result += ".HostMemory(\"" + arg_name + "\")";
}
result += ", " + classname + R"();


REGISTER_OP(")" + name + "\")\n";
for(size_t i = 0; i < tf_scalar_tys.size(); i++){
  std::string arg_name = arg_names[i];
  std::transform(arg_name.begin(), arg_name.end(), arg_name.begin(), [](char c) { return std::tolower(c);});
  result += "  .Input(\"" + arg_name + ": " + tf_scalar_tys[i] + "\")\n";
}
for(size_t i = 0; i < outputs.size(); i++){
  result += "  .Output(\"out" + std::to_string(i) + ": " + tf_scalar_tys[out_idx[i]] + "\")\n";
}
result += ";\n";


  return result;
}


PYBIND11_MODULE(libtriton, m) {
    m.doc() = "Python bindings to the C++ Triton API";
    m.def("make_tensorflow_src", &make_tensorflow_src, "Creates C++ source code for a custom Tensorflow op corresponding to the specified Triton kernel");
}
