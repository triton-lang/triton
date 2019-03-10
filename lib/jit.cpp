#include "triton/jit.h"
#include <string>
#include "triton/ast/ast.h"
#include "triton/ir/context.h"
#include "triton/ir/context_impl.h"
#include "triton/codegen/selection.h"
#include "triton/codegen/tune.h"
#include "triton/codegen/shared_copy.h"
#include "triton/codegen/allocation.h"
#include "triton/codegen/liveness.h"
#include "triton/codegen/vectorize.h"
#include "triton/codegen/buffer_info.h"
#include "triton/codegen/barriers.h"
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



std::unique_ptr<llvm::Module> jit::make_llvm_module(ir::module &module, codegen::tune & tune) {
  llvm::Module* result = new llvm::Module("matmul", llvm_context_);

  // create passes
  codegen::buffer_info_pass buffer_info;
  codegen::place_shared_copy shared(&buffer_info);
  codegen::liveness liveness(&buffer_info);
  codegen::allocation allocation(&liveness, &buffer_info);
  codegen::barriers barriers(&allocation, &buffer_info);
  codegen::vectorize vectorize(&tune);
  codegen::selection selection(&allocation, &tune, &buffer_info);

  // constraints
  std::map<ir::value*, std::vector<std::string>> errors;
  tune.check_constraints(module, errors);
  std::cout << "errors: " << errors.size() << std::endl;
  for(auto &x: errors){
  for(auto &e: x.second)
    std::cout << x.first->get_name() << " " << e << std::endl;
  }
  if(errors.size())
    exit(EXIT_FAILURE);

  // generate ptx
  buffer_info.run(module);
  shared.run(module);
  liveness.run(module);
  allocation.run();
  barriers.run(module);
  vectorize.run(module);
  selection.run(module, *result);

  // launch information
  auto &launch_info_map = launch_info_map_[result->getName()];
  for(unsigned i = 0; i < tune.get_num_global_range(); i++)
    launch_info_map.global_range_size.push_back(tune.get_global_range_size(i));
  launch_info_map.num_threads = tune.get_num_threads();
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


void jit::autotune(ir::module &tt_module, benchmark_t benchmark) {
  // find metaparameters
  codegen::tune tune;
  tune.run(tt_module);
  auto mps = tune.get_params(tt_module);
  // create parameter ranges
  std::vector<std::vector<unsigned>> ranges;
  for(ir::metaparameter *mp: mps){
    std::vector<unsigned> current;
    for(unsigned x = mp->get_lo(); x <= mp->get_hi(); x*=2)
      current.push_back(x);
    ranges.push_back(current);
  }
  // iterate over parameters
  loop_nest<unsigned>(ranges, [&](const std::vector<unsigned> params){
    std::map<ir::value*, std::vector<std::string>> errors;
    unsigned i = 0;
    for(ir::metaparameter *mp: mps)
      mp->set_value(params[i++]);
    tune.check_constraints(tt_module, errors);
    if(errors.size())
      return;
    std::cout << "valid" << std::endl;
  });
}

void jit::autotune(const std::string &src, benchmark_t benchmark) {
  auto ptt_module = make_triton_module(src);
  autotune(*ptt_module, benchmark);
}

void jit::add_module(ir::module &tt_module, const std::vector<unsigned> &params) {
  // set parameters
  codegen::tune tune;
  tune.run(tt_module);
  unsigned i = 0;
  for(ir::metaparameter* mp: tune.get_params(tt_module))
    mp->set_value(params[i++]);
  // compiler to llvm
  auto ll_module = make_llvm_module(tt_module, tune);
  // send llvm module to driver
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
