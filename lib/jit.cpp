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

void jit::init_llvm() {
  static bool init = false;
  if(!init){
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    init = true;
  }
}

std::unique_ptr<llvm::Module> jit::make_llvm_module(ir::module &module, const std::vector<unsigned>& params) {
  llvm::Module* result = new llvm::Module("matmul", llvm_context_);

  // create passes
  codegen::buffer_info_pass buffer_info;
  codegen::place_shared_copy shared(&buffer_info);
  codegen::tune tune;
  codegen::liveness liveness(&buffer_info);
  codegen::allocation allocation(&liveness, &buffer_info);
  codegen::barriers barriers(&allocation, &buffer_info);
  codegen::vectorize vectorize(&tune);
  codegen::selection selection(&allocation, &tune, &buffer_info);

  // tuning parameters
  tune.run(module);
  unsigned i = 0;
  for(ir::metaparameter *x: tune.get_params(module))
    x->set_value(params[i++]);

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

std::string jit::compute_data_layout(bool is_64bit, bool use_short_pointers) {
  std::string ret = "e";
  if (!is_64bit)
    ret += "-p:32:32";
  else if (use_short_pointers)
    ret += "-p3:32:32-p4:32:32-p5:32:32";
  ret += "-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  return ret;
}

void jit::add_module(ir::module &tt_module, const std::vector<unsigned> &params) {
  init_llvm();
  auto ll_module = make_llvm_module(tt_module, params);
  ll_module->setTargetTriple("nvptx64-nvidia-cuda");
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(ll_module->getTargetTriple(), error);
  llvm::TargetMachine *machine = target->createTargetMachine(ll_module->getTargetTriple(), "sm_52", "",
                                                             llvm::TargetOptions(), llvm::Reloc::Model(),
                                                             llvm::None, llvm::CodeGenOpt::Aggressive);
  ll_module->setDataLayout(compute_data_layout());

  // emit machine code
  llvm::legacy::PassManager pass;
  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  machine->addPassesToEmitFile(pass, stream, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
  pass.run(*ll_module);
  std::string src(buffer.begin(), buffer.end());

  modules_.push_back(driver::module(driver_context_, src));
}

void jit::add_module(const std::string &src, const std::vector<unsigned> &params) {
  auto ptt_module = make_triton_module(src);
  add_module(*ptt_module, params);
}

driver::kernel jit::get_function(const std::string &name) {
  return driver::kernel(modules_.front(), name.c_str());
}


}
