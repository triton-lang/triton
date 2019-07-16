#include <string>
#include "triton/lang/lang.h"
#include "triton/codegen/target.h"
#include "triton/ir/context.h"
#include "triton/ir/context_impl.h"
#include "triton/driver/device.h"
#include "triton/driver/error.h"
#include "triton/runtime/jit.h"
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
#include "triton/tools/thread_pool.h"
#include <mutex>

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern triton::lang::translation_unit *ast_root;

namespace triton {
namespace runtime{

void loop_nest(std::vector<size_t> const & ranges,
               std::function<void(std::vector<size_t> const &)> const & f,
               size_t nthreads){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  // thread pools
  ThreadPool pool(nthreads);
  // Start with innermost loop
  size_t i = D - 1;
//  size_t current = 0;
  while(true){
    //Execute function
    pool.enqueue([values, &f](){ f(values); });
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}

template<class T>
void loop_nest(std::vector<std::vector<T>> const & iterates, std::function<void(std::vector<T>)> const & f, size_t nthreads){
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
  loop_nest(ranges, proxy, nthreads);
}




std::unique_ptr<llvm::Module> jit::make_llvm_module(ir::module &module, passes_wrapper &passes, llvm::LLVMContext& llvm_context, launch_information& info) {
  llvm::Module* result = new llvm::Module(module.get_name(), llvm_context);
  passes.selection.run(module, *result);
  // launch information
  info.global_range_size.clear();
  for(unsigned i = 0; i < passes.tune.get_num_global_range(); i++)
    info.global_range_size.push_back(passes.tune.get_global_range_size(i));
  // add globals
  for(auto x: module.globals())
    info.globals[x.first] = ((ir::metaparameter*)x.second)->get_value();
  // number of threads
  info.num_threads = passes.tune.get_num_threads();
  return std::unique_ptr<llvm::Module>(result);
}

triton::lang::translation_unit *jit::parse_program(const char *name, const char *src) {
  // create AST from Triton-C source
  YY_BUFFER_STATE buffer = yy_scan_string(src);
  yyparse();
  yy_delete_buffer(buffer);
  triton::lang::translation_unit *program = ast_root;
  return program;
}

std::unique_ptr<ir::module> jit::make_triton_module(const char * name, triton::ir::context &context, triton::lang::translation_unit *program) {
  // create Triton-IR from AST
  ir::module* module = new ir::module(name, context);
  program->codegen(module);
  return std::unique_ptr<ir::module>(module);
}


jit::jit(driver::context *context, unsigned nthreads): driver_context_(context),
                                    target_(context->device()->make_target()),
                                    nthreads_(nthreads) { }

jit::~jit(){ }

std::vector<unsigned> jit::get_valid(const char *name, const char *src) {
  // find metaparameters
  triton::lang::translation_unit* program = parse_program(name, src);
  auto ptt_module = make_triton_module(name, triton_context_, program);
  ir::module &tt_module = *ptt_module;
  // set parameters
  passes_wrapper passes(target_.get());
  passes.target_independent(tt_module);
  passes.tune.run(tt_module);
  auto mps = passes.tune.get_params(tt_module);
  // create parameter ranges
  std::vector<std::vector<unsigned>> ranges;
  for(ir::metaparameter *mp: mps)
    ranges.push_back(mp->get_space());
  // iterate over parameters
  std::vector<unsigned> result;
  loop_nest<unsigned>(ranges, [&](const std::vector<unsigned> params){
    if(!result.empty())
      return;
    std::map<ir::value*, std::vector<std::string>> errors;
    unsigned i = 0;
    for(ir::metaparameter *mp: mps)
      mp->set_value(params[i++]);
    passes.target_independent(tt_module);
    passes.tune.init(tt_module);
    passes.tune.check_constraints(errors);
//    for(auto e: errors)
//    for(auto x: e.second)
//      std::cout << x << std::endl;
//    std::cout << "-----" << std::endl;
    if(!errors.empty())
      return;
    result = params;
  }, 1);
  if(result.empty())
    throw std::runtime_error("couldn't find valid parameters");
  return result;
}



jit::tune_res_t jit::autotune(const char *name, const char *src, benchmark_t benchmark) {
  // find metaparameters
  triton::lang::translation_unit* program = parse_program(name, src);
  auto ptt_module_0 = make_triton_module(name, triton_context_, program);
  ir::module &tt_module_0 = *ptt_module_0;
  // set parameters
  passes_wrapper passes_0(target_.get());
  passes_0.target_independent(tt_module_0);
  passes_0.tune.run(tt_module_0);
  // create parameter ranges
  std::vector<std::vector<unsigned>> ranges;
  auto mps = passes_0.tune.get_params(tt_module_0);
  for(ir::metaparameter *mp: mps)
    ranges.push_back(mp->get_space());
  // iterate over parameters
  tune_res_t best;
  std::mutex mutex;
  loop_nest<unsigned>(ranges, [&](const std::vector<unsigned> params){
    std::map<ir::value*, std::vector<std::string>> errors;
    unsigned i = 0;
    {
      std::lock_guard<std::mutex> lock(mutex);
      for(ir::metaparameter *mp: mps)
        mp->set_value(params[i++]);
      passes_0.tune.init(tt_module_0);
      passes_0.tune.check_constraints(errors);
    }
    if(!errors.empty())
      return;
    // Deep copy of the module and tuner
    triton::ir::context triton_context;
    auto ptt_module_1 = make_triton_module(name, triton_context, program);
    ir::module &tt_module_1 = *ptt_module_1;
    // run passes
    passes_wrapper passes_1(target_.get());
    passes_1.target_independent(tt_module_1);
    passes_1.tune.run(tt_module_1);
    i = 0;
    for(ir::metaparameter* mp: passes_1.tune.get_params(tt_module_1)){
      mp->set_value(params[i++]);
    }
    passes_1.tune.init(tt_module_1);
    passes_1.target_dependent(tt_module_1);
    driver::device* device = driver_context_->device();
    if(passes_1.shmem_allocation.get_allocated_size() > device->max_shared_memory())
      return;
    if(passes_1.tune.get_num_threads() > device->max_threads_per_block())
      return;
    // Compile
    launch_information info;
    llvm::LLVMContext llvm_context;
    auto ll_module = make_llvm_module(tt_module_1, passes_1, llvm_context, info);
    std::unique_ptr<driver::module> module(driver::module::create(driver_context_, &*ll_module));
    std::unique_ptr<driver::kernel> kernel(driver::kernel::create(module.get(), name));
    double perf;
    {
      std::lock_guard<std::mutex> lock(mutex);
      perf = benchmark(kernel.get(), info);
      if(perf > best.perf){
        best.perf = perf;
        best.params = params;
      }
      for(unsigned p: params)
        std::cout << p << " " << std::flush;
      std::cout << perf << " [ " << best.perf << " ] " << std::endl;
    }
  }, nthreads_);
  std::cout << "Autotuning done - Best performance: " << best.perf << std::endl;
  return best;
}

void jit::add_module(ir::module &tt_module, const std::vector<unsigned> &params) {
  // set parameters
  passes_wrapper passes(target_.get());
  passes.target_independent(tt_module);
  passes.tune.run(tt_module);
  unsigned i = 0;
  for(ir::metaparameter* mp: passes.tune.get_params(tt_module))
    mp->set_value(params[i++]);
  passes.tune.init(tt_module);
  passes.target_dependent(tt_module);
  // check constraints
  std::map<ir::value*, std::vector<std::string>> errors;
  passes.tune.check_constraints(errors);
  for(auto x: errors){
    for(auto str: x.second)
      std::cout <<  x.first->get_name() << ": " << str << std::endl;
  }
  if(errors.size())
    throw std::runtime_error("invalid parameters");
  // triton module -> llvm module
  std::string name = tt_module.get_name();
  auto ll_module = make_llvm_module(tt_module, passes, llvm_context_, launch_info_map_[name]);
  // llvm module -> machine code
  modules_.insert({name, driver::module::create(driver_context_, &*ll_module)});
}

void jit::add_module(const char *name, const char *src, const std::vector<unsigned> &params) {
  triton::lang::translation_unit* program = parse_program(name, src);
  auto ptt_module = make_triton_module(name, triton_context_, program);
  add_module(*ptt_module, params);
}

driver::kernel *jit::get_function(const char *name) {
  return driver::kernel::create(modules_.at(name), name);
}

launch_information jit::get_launch_info(const char *name) {
  return launch_info_map_.at(name);
}


}
}
