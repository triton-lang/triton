#include <string>
#include <mutex>
#include <regex>
#include <functional>
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
#include "llvm/IR/Module.h"


typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern triton::lang::translation_unit *ast_root;

namespace triton{
namespace runtime {


// helpers
void _parallel_loop_nest(std::vector<size_t> const & ranges,
                       std::function<void(std::vector<size_t> const &)> const & f,
                       size_t nthreads){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  // Start with innermost loop
  size_t i = D - 1;
  while(true){
    //  Execute function
    f(values);
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}

template<class T>
void _parallel_loop_nest(std::vector<std::vector<T>> const & iterates, std::function<void(std::vector<T>)> const & f, size_t nthreads){
  //Ranges to iterate over
  std::vector<size_t> ranges;
  for(auto const & x: iterates)
    ranges.push_back(x.size());
  //Proxy function
  auto proxy = [&](std::vector<size_t> const & idx){
    std::vector<T> x(iterates.size());
    for(size_t i = 0; i < x.size(); ++i)
    x[i] = iterates[i][idx[i]];
  f(x);
  };
  //Iterate
  _parallel_loop_nest(ranges, proxy, nthreads);
}

// caller

arg_type convert(ir::type *ty) {
  if(ty->is_integer_ty(1))
    return INT1_T;
  if(ty->is_integer_ty(8))
    return INT8_T;
  if(ty->is_integer_ty(16))
    return INT16_T;
  if(ty->is_integer_ty(32))
    return INT32_T;
  if(ty->is_integer_ty(64))
    return INT64_T;
  if(ty->is_half_ty())
    return HALF_T;
  if(ty->is_float_ty())
    return FLOAT_T;
  if(ty->is_double_ty())
    return DOUBLE_T;
  if(ty->is_pointer_ty())
    return BUFFER_T;
  throw std::runtime_error("unknown type");
}

function::caller::caller(ir::function *ir, std::shared_ptr<driver::module> parent, size_t n_threads)
  : bin_(driver::kernel::create(&*parent, ir->get_name().c_str())), n_threads_(n_threads), parent_(parent) {
  // extract signature
  ir::function_type* ty = ir->get_fn_type();
  for(int i = 0; i < ty->get_num_params(); i++)
    param_tys_.push_back(convert(ty->get_param_ty(i)));
}


void function::caller::operator ()(driver::stream *stream, const std::array<size_t, 3>& grid, const std::vector<arg>& args) const {
  if(args.size() != param_tys_.size())
    throw std::runtime_error("invalid number of arguments");
  for(size_t i = 0; i < args.size(); i++){
    arg arg_i = args.at(i);
    arg_type ty = arg_i.type();
    if(ty != param_tys_.at(i))
      throw std::runtime_error("invalid type");
    if(ty == BUFFER_T)
      bin_->setArg(i, *((driver::buffer**)arg_i.data()));
    else
      bin_->setArg(i, size_of(ty), arg_i.data());
  }
  stream->enqueue(&*bin_, grid, {n_threads_, 1, 1});
}



// module
triton::lang::translation_unit *function::make_ast(const char *src) {
  YY_BUFFER_STATE buffer = yy_scan_string(src);
  yyparse();
  yy_delete_buffer(buffer);
  triton::lang::translation_unit *program = ast_root;
  return program;
}

std::unique_ptr<ir::module> function::make_ir(triton::lang::translation_unit *program) {
  // create Triton-IR from AST
  ir::module* module = new ir::module("", ctx_);
  program->codegen(module);
  return std::unique_ptr<ir::module>(module);
}

options function::autotune(lang::translation_unit *ast, driver::stream* stream, const grid_fn_ty& grid_fn, const std::vector<arg>& args) {
  std::unique_ptr<ir::module> ir = make_ir(ast);
  // extract tunable values
  std::vector<std::pair<std::string, ir::metaparameter*>> values;
  for(auto it: ir->globals())
  if(auto *mp = dynamic_cast<ir::metaparameter*>(it.second))
    values.push_back({it.first, mp});
  // extract search space
  std::vector<std::vector<unsigned>> space;
  space.push_back({1, 2, 4, 8}); // num warps
  for(auto it: values)
    space.push_back(it.second->get_space());
  // exhaustive search
  struct profile_t{
    double ts;
    std::vector<unsigned> params;
  };
  profile_t best = { INFINITY };
  std::function<void(std::vector<unsigned>)> benchmark =
      [&](std::vector<unsigned> params) {
    // options
    options opt;
    unsigned i = 0;
    opt.num_warps = params[i++];
    for(auto it: values)
      opt.params[it.first] = params[i++];
    // make binary
    auto ir = make_ir(ast);
    auto bin = make_bin(*ir, stream->context(), opt);
    // benchmark
    ir::function *tmp = ir->get_function_list()[0];
    caller fn(tmp, std::move(bin), opt.num_warps * 32);
    double ts = tools::bench([&]() { fn(stream, grid_fn(opt.params), args); }, stream);
    if(ts < best.ts)
      best = {ts, params};
  };
  _parallel_loop_nest<unsigned>(space, benchmark, 1);
  // populate options
  unsigned current = 0;
  options opt;
  opt.num_warps = best.params[current++];
  for(auto it: values)
    opt.params[it.first] = best.params[current++];
  return opt;
}


std::unique_ptr<driver::module> function::make_bin(ir::module &module, driver::context *context, const options& opt) {
  std::unique_ptr<codegen::target> target = context->device()->make_target();
  // update metaparameter values
  for(auto x: opt.params)
  if(auto* mp = dynamic_cast<ir::metaparameter*>(module.globals().at(x.first)))
    mp->set_value(x.second);
  // create passes
  codegen::analysis::tune tune(opt.num_warps);
  codegen::analysis::shmem::info shmem_info;
  codegen::analysis::shmem::liveness shmem_liveness(&shmem_info);
  codegen::analysis::shmem::allocation shmem_allocation(&shmem_liveness, &shmem_info, &tune);
  codegen::analysis::alignment_info alignment_info;
  codegen::transform::shmem_barriers shmem_barriers(&shmem_allocation, &shmem_info);
  codegen::transform::vectorize vectorize(&tune);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole;
  codegen::transform::reassociate reassociate(&tune);
  codegen::selection selection(&shmem_allocation, &tune, &shmem_info, &alignment_info, target.get());
  // run passes
  peephole.run(module);
  dce.run(module);
  tune.run(module);
  tune.init(module);
  reassociate.run(module);
  peephole.run(module);
  if(target->is_gpu()){
    shmem_info.run(module);
    shmem_liveness.run(module);
    shmem_allocation.run();
    shmem_barriers.run(module);
  }
  alignment_info.run(module);
  vectorize.run(module);
  dce.run(module);
  // generate llvm code
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(module.get_name(), ctx));
  selection.run(module, *llvm);
  // return binary
  std::unique_ptr<driver::module> res(driver::module::create(context, llvm.get()));
  return res;
}


function::function(const std::string &src): src_(src) {
  // src -> ast
  ast_ = make_ast(src_.c_str());
}

void function::operator()(const std::vector<arg>& args, const grid_fn_ty& grid_fn, driver::stream *stream) {
  /* determine if should re-tune or not */
  cache_key_t key;
  // re-tune if device is difference
  key.first = stream->context()->device();
  // re-tune if any int argument is different
  for(size_t i = 0; i < args.size(); i++){
    arg_type ty = args.at(i).type();
    if(is_int_type(ty)){
      long val = 0;
      std::memcpy((void*)&val, args.at(i).data(), size_of(ty));
      key.second.push_back(val);
    }
  }

  /* find existing configuration */
  auto it = cache_.find(key);
  if(it != cache_.end()){
    it->second.second(stream, grid_fn(it->second.first.params), args);
    return;
  }

  /* re-tune and re-compile */
  options opt = autotune(ast_, stream, grid_fn, args);
  std::unique_ptr<ir::module> ir = make_ir(ast_);
  std::unique_ptr<driver::module> bin = make_bin(*ir, stream->context(), opt);
  ir::function* fn = ir->get_function_list().front();
  const caller& run = cache_.insert({key, cache_val_t{opt, caller(fn, std::move(bin), opt.num_warps*32)}}).first->second.second;
  run(stream, grid_fn(opt.params), args);
}

void function::operator()(const std::vector<arg>& args, const grid_t& grid, driver::stream *stream) {
  return this->operator()(args, [&grid](const params_t&){ return grid; }, stream);
}

std::string to_tf_ty(ir::type *ty) {
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

std::string ref_to_tf_ty(ir::type *ty) {
  std::string res = to_tf_ty(ty);
  if(ty->is_pointer_ty())
    res = "const " + res + "&";
  return res;
}


std::string function::make_tensorflow_src(const std::vector<size_t>& outputs, const std::string& macro) {
  std::unique_ptr<ir::module> ir = make_ir(ast_);
  // extract function signature
  ir::function* fn = ir->get_function_list().front();
  ir::function_type* fn_ty = fn->get_fn_type();
  // numberof arguments
  size_t n_args = fn_ty->get_num_params();
  size_t n_outputs = outputs.size();
  // extract function name
  std::string name = fn->get_name();
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
  std::vector<std::string> tf_tys;
  std::transform(fn_ty->params_begin(), fn_ty->params_end(), std::back_inserter(tf_tys), to_tf_ty);
  std::vector<std::string> tf_cref_tys;
  std::transform(fn_ty->params_begin(), fn_ty->params_end(), std::back_inserter(tf_cref_tys), ref_to_tf_ty);

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

std::string src = R"TTKERNSRC( )" + src_ + ")TTKERNSRC\";" + R"(

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
    context->set_output()" + str_i[i] + ", " + arg_names[outputs[i]] + ");";

result += R"(

    // wrap tensors)";
for(size_t i: ptr_idx)
result += R"(
    drv::cu_buffer cu_)" + arg_names[i] + "(ctx, " + arg_names[i] + ".tensor_data().size(), (CUdeviceptr)" + arg_names[i] + R"(.tensor_data().data(), false);)";


std::regex regex("#([a-zA-Z]([a-zA-Z]|[0-9])*)");
std::string grid_str = std::regex_replace(macro, regex, "x.at(\"$1\")");

result += R"(

    // create launch grid;
    auto grid = [&](const rt::params_t& x) { return rt::grid_t{)" + grid_str + R"(}; };)";

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

REGISTER_KERNEL_BUILDER(Name(")" + name + "\").Device(DEVICE_GPU), " + classname + R"();

REGISTER_OP(")" + name + "\")\n";
for(size_t i = 0; i < tf_tys.size(); i++){
  bool is_output = std::find(outputs.begin(), outputs.end(), i) != outputs.end();
  std::string mode = is_output ? "Output" : "Input" ;
  result += "  ." + mode + "(\"" + arg_names[i] + ": " + tf_tys[i] + "\")\n";
}
result += ";\n";


  return result;
}




}
}
