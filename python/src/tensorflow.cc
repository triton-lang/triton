#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
#include <regex>
#include <algorithm>
#include "triton/codegen/selection/selection.h"
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


/* TF triton op properties */

std::map<size_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
std::map<size_t, std::shared_ptr<rt::function>> id_fn_map;
std::map<size_t, int64_t> i64scalar_map;

void register_grid(size_t id,
                   const rt::function::grid_fn_ty& grid_fn) {
  id_grid_map[id].reset(new rt::function::grid_fn_ty(grid_fn));
}

void register_fn(size_t id,
                   const std::string& src,
                   const rt::function::options_space_t& opt) {
  id_fn_map[id].reset(new rt::function(src, opt));
}

size_t make_op_id() {
  return id_fn_map.size();
}

size_t make_scalar_id() {
  return i64scalar_map.size();
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

void gen_make_launch_function(std::ostream &os, const std::vector<ir::argument*>& args) {
  os << "  (*id_fn_map.at(id_))({";
  for(unsigned i = 0; i < args.size() ; i++){
    ir::argument *arg = args[i];
    std::string name = arg->get_name();
    if(arg->get_type()->is_pointer_ty())
      name = "&cu_" + name;
    if(i > 0)
      os << ", ";
    os << name;
  }
  os << "}, *id_grid_map.at(id_), stream);  \n";
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
    os << "  .Attr(\"T" << i << " : {bool, int8, int16, int32, int64, float16, float32, float64}\")" << std::endl;
    os << "  .Input(\"" << name << ": T" << i << "\")\n";
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
  os << "  .Attr(\"id: int\")" << std::endl;
  os << ";\n";
}

inline std::string preheader() {
return
R"(
#define bool _Bool
#define true 1
#define false 0
#define __bool_true_false_are_defined 1

#define __readonly      __attribute__((readonly))
#define __writeonly     __attribute__((writeonly))
#define __noalias       __attribute__((noalias))
#define __aligned(A)    __attribute__((aligned(A)))
#define __multipleof(A) __attribute__((multipleof(A)))

extern int get_program_id(int);
)";
}

std::tuple<std::string,
           std::string> make_tensorflow_src(std::string src,
                                const std::vector<std::string>& outputs,
                                const runtime::function::options_space_t& opt)
{
  src = preheader() + src;
  // pre-process
  TokenSequence tokens;
  Preprocessor cpp(&src, true);
  for(auto it: opt.defines){
    cpp.AddMacro(it.first, &it.second[0]);
  }
  cpp.Process(tokens);
  // parse
  Parser parser(tokens);
  parser.Parse();
  // triton-ir code-gen
  ir::context ctx;
  auto ir = std::shared_ptr<ir::module>(new ir::module("", ctx));
  Generator gen(&parser);
  gen.Gen(&*ir);
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

extern std::map<size_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
extern std::map<size_t, std::shared_ptr<rt::function>> id_fn_map;


class )" << opname << R"(: public OpKernel {
 public:
  explicit )" << opname << R"((OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
  }

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
    )";
oss << R"(
    // launch function
    )";
gen_make_launch_function(oss, fn->args());
oss << R"(
  }

private:
  int id_;
};

// register kernel builder
)";
gen_register_kernel_builder(oss, cc_name, opname, fn->args());
oss << R"(
// register op
)";
gen_register_op(oss, cc_name, fn->args(), outputs);


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
    m.def("register_fn", &register_fn);
    m.def("make_op_id", &make_op_id);
    m.def("make_scalar_id", &make_scalar_id);
    m.def("retrieve_scalar", &retrieve_scalar)
    ;
}
