#include "triton/jit.h"
#include <string>
#include "triton/ast/ast.h"
#include "triton/ir/context.h"
#include "triton/ir/context_impl.h"
#include "triton/driver/device.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Analysis/LoopPass.h"

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
using triton::ast::translation_unit;
extern translation_unit *ast_root;

namespace triton {

void loop_nest(std::vector<size_t> const & ranges, std::function<void(std::vector<size_t> const &)> const & f){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  // Start with innermost loop
  size_t i = D - 1;
  while(true){
    //Execute function
    f(values);
    //Increment counters
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}

template<class T>
void loop_nest(std::vector<std::vector<T>> const & iterates, std::function<void(std::vector<T>)> const & f){
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
  loop_nest(ranges, proxy);
}




std::unique_ptr<llvm::Module> jit::make_llvm_module(ir::module &module, passes_wrapper &passes) {
  llvm::Module* result = new llvm::Module("matmul", llvm_context_);
  passes.selection.run(module, *result);
  // launch information
  auto &launch_info_map = launch_info_map_[result->getName()];
  for(unsigned i = 0; i < passes.tune.get_num_global_range(); i++)
    launch_info_map.global_range_size.push_back(passes.tune.get_global_range_size(i));
  launch_info_map.num_threads = passes.tune.get_num_threads();
  return std::unique_ptr<llvm::Module>(result);
}

std::unique_ptr<ir::module> jit::make_triton_module(const std::string &src) {
  // create AST from Triton-C source
  YY_BUFFER_STATE buffer = yy_scan_string(src.c_str());
  yyparse();
  yy_delete_buffer(buffer);
  translation_unit *program = ast_root;
  // create Triton-IR from AST
  ir::module* module = new ir::module("matrix", triton_context_);
  program->codegen(module);
  return std::unique_ptr<ir::module>(module);
}


jit::jit(driver::context context): driver_context_(context) {
}


void jit::autotune(const std::string &src, benchmark_t benchmark) {
  // find metaparameters
  auto ptt_module = make_triton_module(src);
  ir::module &tt_module = *ptt_module;
  // set parameters
  passes_wrapper passes;
  passes.tune.run(tt_module);
  auto mps = passes.tune.get_params(tt_module);
  // create parameter ranges
  std::vector<std::vector<unsigned>> ranges;
  for(ir::metaparameter *mp: mps){
    std::vector<unsigned> current;
    for(unsigned x = mp->get_lo(); x <= mp->get_hi(); x*=2)
      current.push_back(x);
    ranges.push_back(current);
  }
  // iterate over parameters
  unsigned i;
  loop_nest<unsigned>(ranges, [&](const std::vector<unsigned> params){
    std::map<ir::value*, std::vector<std::string>> errors;
    i = 0;
    for(ir::metaparameter *mp: mps)
      mp->set_value(params[i++]);
    passes.tune.init(tt_module);
    if(!passes.tune.check_constraints(errors))
      return;
    // Deep copy of the module and tuner
    auto ptt_module = make_triton_module(src);
    ir::module &tt_module = *ptt_module;
    passes_wrapper passes;
    passes.tune.run(tt_module);
    i = 0;
    for(ir::metaparameter* mp: passes.tune.get_params(tt_module)){
      mp->set_value(params[i++]);
    }
    passes.tune.init(tt_module);
    passes.init(tt_module);
    const driver::device &device = driver_context_.device();
    if(passes.allocation.get_allocated_size() > device.max_shared_memory())
      return;
    if(passes.tune.get_num_threads() > device.max_threads_per_block())
      return;
    // Compile
    auto ll_module = make_llvm_module(tt_module, passes);
    driver::module module(driver_context_, &*ll_module);
    driver::kernel kernel(module, "matmul");
    launch_information info = launch_info_map_.at("matmul");
    for(unsigned p: params)
      std::cout << p << " " << std::flush;
    benchmark(kernel, info);
  });
}

void jit::add_module(ir::module &tt_module, const std::vector<unsigned> &params) {
  // set parameters
  passes_wrapper passes;
  passes.tune.run(tt_module);
  unsigned i = 0;
  for(ir::metaparameter* mp: passes.tune.get_params(tt_module))
    mp->set_value(params[i++]);
  passes.tune.init(tt_module);
  passes.init(tt_module);
  // check constraints
  std::map<ir::value*, std::vector<std::string>> errors;
  passes.tune.check_constraints(errors);
  if(errors.size())
    throw std::runtime_error("invalid parameters");
  if(passes.allocation.get_allocated_size() > driver_context_.device().max_shared_memory())
    throw std::runtime_error("invalid parameters");
  // triton module -> llvm module
  auto ll_module = make_llvm_module(tt_module, passes);
  // llvm module -> machine code
  modules_.push_back(driver::module(driver_context_, &*ll_module));
}

void jit::add_module(const std::string &src, const std::vector<unsigned> &params) {
  auto ptt_module = make_triton_module(src);
  add_module(*ptt_module, params);
}

driver::kernel jit::get_function(const std::string &name) {
  return driver::kernel(modules_.front(), name.c_str());
}

jit::launch_information jit::get_launch_info(const std::string &name) {
  return launch_info_map_.at(name);
}

}
