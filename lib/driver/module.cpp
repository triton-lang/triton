/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <iostream>
#include <fstream>

#include "triton/driver/module.h"
#include "triton/driver/context.h"
#include "triton/driver/error.h"
#include "triton/tools/sys/getenv.hpp"
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

namespace triton
{
namespace driver
{

/* ------------------------ */
//         Base             //
/* ------------------------ */

void module::init_llvm() {
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

module::module(driver::context* ctx, CUmodule mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), ctx_(ctx) {
}

module::module(driver::context* ctx, cl_program mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), ctx_(ctx) {
}

driver::context* module::context() const {
  return ctx_;
}

module* module::create(driver::context* ctx, llvm::Module *src) {
  if(dynamic_cast<driver::cu_context*>(ctx))
    return new cu_module(ctx, src);
  if(dynamic_cast<driver::ocl_device*>(ctx))
    return new ocl_module(ctx, src);
  throw std::runtime_error("unknown context");
}

void module::compile_llvm_module(llvm::Module* module, const std::string& triple,
                                        const std::string &proc, std::string layout,
                                        llvm::SmallVectorImpl<char> &buffer) {
  init_llvm();
  // create machine
  module->setTargetTriple(triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), proc, "",
                                                             llvm::TargetOptions(), llvm::Reloc::Model(),
                                                             llvm::None, llvm::CodeGenOpt::Aggressive);


  // set data layout
  if(layout.empty())
    layout = module->getDataLayoutStr();
  module->setDataLayout(layout);

  // emit machine code
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);
  machine->addPassesToEmitFile(pass, stream, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
  pass.run(*module);
}

/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

ocl_module::ocl_module(driver::context * context, llvm::Module* src): module(context, cl_program(), true) {
  init_llvm();
  llvm::SmallVector<char, 0> buffer;
  module::compile_llvm_module(src, "amdgcn-amd-amdpal", "gfx902", "", buffer);
  throw std::runtime_error("need to implement opencl module creation");
}


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

std::string cu_module::compile_llvm_module(llvm::Module* module) {
  // set data layout
  std::string layout = "e";
  bool is_64bit = true;
  bool use_short_pointers = true;
  if (!is_64bit)
    layout += "-p:32:32";
  else if (use_short_pointers)
    layout += "-p3:32:32-p4:32:32-p5:32:32";
  layout += "-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  // create
  llvm::SmallVector<char, 0> buffer;
  module::compile_llvm_module(module, "nvptx64-nvidia-cuda", "sm_52", layout, buffer);
  return std::string(buffer.begin(), buffer.end());
}



cu_module::cu_module(driver::context * context, llvm::Module* ll_module): cu_module(context, compile_llvm_module(ll_module)) { }

cu_module::cu_module(driver::context * context, std::string const & source) : module(context, CUmodule(), true), source_(source){
  cu_context::context_switcher ctx_switch(*context);
  // JIT compile source-code
  CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
  unsigned int errbufsize = 8096;
  std::string errbuf(errbufsize, 0);
  void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)errbuf.data()};
  try{
    dispatch::cuModuleLoadDataEx(&*cu_, source_.data(), 2, opt, optval);
  }catch(exception::cuda::base const &){
    std::cerr << "Compilation Failed! Log: " << std::endl;
    std::cerr << errbuf << std::endl;
    throw;
  }
}

cu_buffer cu_module::symbol(const char *name) const{
  CUdeviceptr handle;
  size_t size;
  dispatch::cuModuleGetGlobal_v2(&handle, &size, *cu_, name);
  return cu_buffer(ctx_, handle, false);
}


}
}

