#include <string>
#include <mutex>
#include <regex>
#include <functional>
#include <algorithm>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/tiles.h"
#include "triton/codegen/selection.h"
#include "triton/runtime/function.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/lang/cpp.h"
#include "triton/lang/parser.h"
#include "triton/lang/code_gen.h"
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/driver/error.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/print.h"
#include "triton/tools/bench.hpp"
#include "llvm/IR/Module.h"




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

function::caller::caller(ir::function *ir, std::shared_ptr<driver::module> parent, const options_t& opt)
  : bin_(driver::kernel::create(&*parent, ir->get_name().c_str())), parent_(parent), opt_(opt) {
  // extract signature
  ir::function_type* ty = ir->get_fn_type();
  for(size_t i = 0; i < ty->get_num_params(); i++)
    param_tys_.push_back(convert(ty->get_param_ty(i)));
}


void function::caller::operator ()(driver::stream *stream, const grid_t& _grid, const std::vector<arg>& args) const {
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
  // sanity check
  if(_grid.size() > 3)
    throw std::runtime_error("grid size must be no greater than 3");
  std::array<size_t, 3> grid;
  for(size_t i = 0; i < 3; i++)
    grid[i] = (i < _grid.size()) ? _grid[i] : 1;
  stream->enqueue(&*bin_, grid, {opt_.num_warps * 32, 1, 1});
}


std::unique_ptr<ir::module> function::make_ir(Parser& parser) {
  // create Triton-IR from AST
  ir::module* module = new ir::module("", ctx_);
  Generator gen(&parser);
  gen.Gen(module);
  return std::unique_ptr<ir::module>(module);
}


function::caller function::autotune(driver::stream* stream, const grid_fn_ty& grid_fn,
                                       const std::vector<arg>& args) {

  // all tuning parameters are strings
  std::vector<std::string> num_warps;
  for(size_t i: opt_space_.num_warps)
    num_warps.push_back(std::to_string(i));
  std::vector<std::vector<std::string>> space;
  space.push_back(num_warps);
  for(const auto& i: opt_space_.defines)
    space.push_back(i.second);

  // exhaustive search
  double best_ts = INFINITY;
  std::unique_ptr<caller> ret;

  auto benchmark = [&](std::vector<std::string> params) {
    // extract options
    options_t opt;
    unsigned i = 0;
    opt.num_warps = std::stoi(params[i++]);
    for(auto it: opt_space_.defines){
      opt.defines[it.first] = params[i++];
    }
    // pre-process
    TokenSequence tokens;
    Preprocessor cpp(&src_, true);
    for(auto it: opt_space_.defines)
      cpp.AddMacro(it.first, &opt.defines.at(it.first));
    cpp.Process(tokens);
//    tokens.Print(stdout);
    // parse
    Parser parser(tokens);
    parser.Parse();
    // triton-ir code-gen
    auto ir = make_ir(parser);
    // binary code-gen
    std::unique_ptr<driver::module> bin;
    try{
      bin = make_bin(*ir, stream->context(), opt);
    }catch(const std::runtime_error& e) {
      return;
    }
    // kernel uses too much resources
    if(!bin)
      return;
    // benchmark
    ir::function *tmp = ir->get_function_list()[0];
    caller call(tmp, std::move(bin), opt);
    double ts = tools::bench([&]() { call(stream, grid_fn(opt), args); }, stream);
    // save best
    if(ts < best_ts) {
      best_ts = ts;
      ret.reset(new caller(call));
    }
  };
  _parallel_loop_nest<std::string>(space, benchmark, 1);
  if(!ret)
    throw std::runtime_error("could not find valid option in provided space");
  return *ret;
}


std::unique_ptr<driver::module> function::make_bin(ir::module &module, driver::context *context, const options_t& opt) {
  std::unique_ptr<codegen::target> target = context->device()->make_target();
  // generate llvm code
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(module.get_name(), ctx));
  // create passes
  codegen::analysis::align align;
  codegen::analysis::axes axes;
  codegen::analysis::layout layouts(&axes, &align);
  codegen::analysis::tiles tiles(opt.num_warps, &align, &axes, &layouts);
  codegen::analysis::liveness liveness(&tiles, &layouts);
  codegen::analysis::allocation allocation(&liveness, &tiles);
  codegen::transform::membar barriers(&liveness, &allocation);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole;
  codegen::transform::reassociate reassociate(&align);
  codegen::transform::coalesce coalesce(&align, &layouts);
  codegen::transform::cts cts;
  codegen::selection selection(&liveness, &allocation, &tiles, &align, &axes, &layouts, target.get(), opt.num_warps);
  // run passes
  peephole.run(module);
  dce.run(module);
  align.run(module);
  cts.run(module);
  axes.run(module);
  layouts.run(module);
  coalesce.run(module);
  dce.run(module);
  align.run(module);
  dce.run(module);
  reassociate.run(module);
//  ir::print(module, std::cout);
//  exit(EXIT_FAILURE);
  dce.run(module);
  cts.run(module);
  align.run(module);
  axes.run(module);
  layouts.run(module);
  tiles.run(module);
  liveness.run(module);
  allocation.run(module);
  if(allocation.allocated_size() > context->device()->max_shared_memory())
    return std::unique_ptr<driver::module>();
  barriers.run(module);
  dce.run(module);
  axes.run(module);
  layouts.run(module);
//  ir::print(module, std::cout);
  align.run(module);
  tiles.run(module);
  selection.run(module, *llvm);
  // return binary
  std::unique_ptr<driver::module> res(driver::module::create(context, std::move(llvm)));
  // done
  return res;
}

std::string preheader() {
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

function::function(const std::string &src, const options_space_t& opt):  src_(src), opt_space_(opt) {
  src_ = preheader() + src_;
}

void function::operator()(const std::vector<arg>& args, const grid_fn_ty& grid_fn, driver::stream *stream) {
  cache_key_t key;

  /* figure out if the kernel should be re-tuned */
  // re-tune if device is different
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
    it->second(stream, grid_fn(it->second.opt()), args);
    return;
  }

  /* re-tune and re-compile */
  cache_.insert({key, autotune(stream, grid_fn, args)});
}

void function::operator()(const std::vector<arg>& args, const grid_t& grid, driver::stream *stream) {
  return this->operator()(args, [&grid](const options_t&){ return grid; }, stream);
}

}
}
