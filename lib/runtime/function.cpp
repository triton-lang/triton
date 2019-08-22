#include <string>
#include <mutex>
#include <regex>
#include <functional>
#include <algorithm>
#include "triton/codegen/selection/selection.h"
#include "triton/runtime/function.h"
#include "triton/lang/lang.h"
#include "triton/lang/wgtcc/cpp.h"
#include "triton/lang/wgtcc/parser.h"
#include "triton/lang/wgtcc/code_gen.h"
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
  : bin_(driver::kernel::create(&*parent, ir->get_name().c_str())), parent_(parent), n_threads_(n_threads) {
  // extract signature
  ir::function_type* ty = ir->get_fn_type();
  for(size_t i = 0; i < ty->get_num_params(); i++)
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
triton::lang::translation_unit *function::make_ast(const char *csrc) {
  std::string src(csrc);
  Preprocessor cpp(&src, true);
//  for (auto& def: defines)
//    DefineMacro(cpp, def);
//  for (auto& path: include_paths)
//    cpp.AddSearchPath(path);

  FILE* fp = stdout;
//  if (specified_out_name) {
//    fp = fopen(filename_out.c_str(), "w");
//  }
  TokenSequence ts;
  cpp.Process(ts);
  Parser parser(ts);
  parser.Parse();
  Generator gen(&parser);
  ir::module out("", ctx_);
  gen.Gen(&out);
  exit(EXIT_FAILURE);

//  if (only_preprocess) {
//    ts.Print(fp);
//    return 0;
//  }

  YY_BUFFER_STATE buffer = yy_scan_string(csrc);
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
  profile_t best = { INFINITY, {} };
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
  codegen::analysis::grids tune(opt.num_warps);
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

}
}
