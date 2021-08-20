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
#include <fstream>
#include <unistd.h>
#include <memory>
#include <regex>
#ifdef __HIP_PLATFORM_AMD__
#include "triton/driver/module_hip.h"
#include "triton/driver/context_hip.h"
#else
#include "triton/driver/module.h"
#include "triton/driver/context.h"
#endif
#include "triton/driver/error.h"
#include "triton/tools/sha1.hpp"
#include "triton/tools/sys/getenv.hpp"
#include "triton/tools/sys/mkdir.hpp"
#include "triton/tools/sys/exec.hpp"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Transforms/Utils/Cloning.h"
#ifdef __HIP_PLATFORM_AMD__
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "triton/tools/rocm_helper.h"
#endif

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

#ifdef __HIP_PLATFORM_AMD__
  void LLVMInitializeAMDGPUTargetInfo();
  void LLVMInitializeAMDGPUTarget();
  void LLVMInitializeAMDGPUTargetMC();
  void LLVMInitializeAMDGPUAsmPrinter();
  void LLVMInitializeAMDGPUAsmParser();
#else
  void LLVMInitializeNVPTXTargetInfo();
  void LLVMInitializeNVPTXTarget();
  void LLVMInitializeNVPTXTargetMC();
  void LLVMInitializeNVPTXAsmPrinter();
  void LLVMInitializeNVPTXAsmParser();
#endif


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
#ifdef __HIP_PLATFORM_AMD__
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
#else
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
#endif 
    init = true;
  }
}

module::module(CUmodule mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), spilled_(0) {
}

module::module(host_module_t mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), spilled_(0) {
}


module* module::create(driver::device* device, std::unique_ptr<llvm::Module> src) {
  switch(device->backend()){
    case CUDA: return new cu_module(device, std::move(src));
    case Host: return new host_module(std::move(src));
    default: throw std::runtime_error("unknown backend");
  }
}

void module::compile_llvm_module(std::unique_ptr<llvm::Module> module, const std::string& triple,
                                 const std::string &proc, std::string layout,
                                 llvm::SmallVectorImpl<char> &buffer,
                                 const std::string& features,
                                 file_type_t ft) {

}


/* ------------------------ */
//        Host              //
/* ------------------------ */

host_module::host_module(std::unique_ptr<llvm::Module> src): module(host_module_t(), true) {
  throw std::runtime_error("CPU unsupported");
//  init_llvm();
//  // create kernel wrapper
//  llvm::LLVMContext &ctx = src->getContext();
//  llvm::Type *void_ty = llvm::Type::getVoidTy(ctx);
//  llvm::Type *args_ty = llvm::Type::getInt8PtrTy(ctx)->getPointerTo();
//  llvm::Type *int32_ty = llvm::Type::getInt32Ty(ctx);
//  std::vector<llvm::Type*> tys = {args_ty, int32_ty, int32_ty, int32_ty};
//  llvm::FunctionType *main_ty = llvm::FunctionType::get(void_ty, tys, false);
//  llvm::Function* main = llvm::Function::Create(main_ty, llvm::Function::ExternalLinkage, "_main", &*src);
//  llvm::Function* fn = &*src->getFunctionList().begin();
//  llvm::FunctionType *fn_ty = fn->getFunctionType();
//  std::vector<llvm::Value*> fn_args(fn_ty->getNumParams());
//  std::vector<llvm::Value*> ptrs(fn_args.size() - 3);
//  llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", main);
//  llvm::IRBuilder<> ir_builder(ctx);
//  ir_builder.SetInsertPoint(entry);
//  auto get_size = [](llvm::Type* ty) { return ty->isPointerTy() ? sizeof(char*) : ty->getPrimitiveSizeInBits() / 8; };
//  llvm::Value* base = main->arg_begin();
//  llvm::Value* args_base = ir_builder.CreateBitCast(base, base->getType()->getPointerElementType());

//  size_t offset = 0;
//  for(unsigned i = 0; i < ptrs.size(); i++){
//    ptrs[i] = ir_builder.CreateGEP(args_base, ir_builder.getInt32(offset));
//    size_t nbytes = get_size(fn_ty->getParamType(i));
//    offset += nbytes;
//    if(i < ptrs.size() - 1){
//      size_t np1bytes = get_size(fn_ty->getParamType(i+1));
//      offset = (offset + np1bytes - 1) / np1bytes * np1bytes;
//    }
//  }
//  for(unsigned i = 0; i < ptrs.size(); i++)
//    ptrs[i] = ir_builder.CreateBitCast(ptrs[i], fn_ty->getParamType(i)->getPointerTo());
//  for(unsigned i = 0; i < ptrs.size(); i++)
//    fn_args[i] = ir_builder.CreateLoad(ptrs[i]);

//  fn_args[fn_args.size() - 3] = main->arg_begin() + 1;
//  fn_args[fn_args.size() - 2] = main->arg_begin() + 2;
//  fn_args[fn_args.size() - 1] = main->arg_begin() + 3;
//  ir_builder.CreateCall(fn, fn_args);
//  ir_builder.CreateRetVoid();

////  llvm::legacy::PassManager pm;
////  pm.add(llvm::createPrintModulePass(llvm::outs()));
////  pm.add(llvm::createVerifierPass());
////  pm.run(*src);

////   create execution engine
//  for(llvm::Function& fn: src->functions())
//    hst_->functions[fn.getName().str()] = &fn;

////  llvm::orc::JITTargetMachineBuilder JTMB = *llvm::orc::JITTargetMachineBuilder::detectHost();
////  auto DL = JTMB.getDefaultDataLayoutForTarget();
////  auto CIRC = std::unique_ptr<llvm::orc::ConcurrentIRCompiler>(new llvm::orc::ConcurrentIRCompiler(JTMB));
////  hst_->ES = new llvm::orc::ExecutionSession();
////  hst_->ObjectLayer = new llvm::orc::RTDyldObjectLinkingLayer(*hst_->ES, []() { return std::unique_ptr<llvm::SectionMemoryManager>(new llvm::SectionMemoryManager()); });
////  hst_->CompileLayer = new llvm::orc::IRCompileLayer(*hst_->ES, *hst_->ObjectLayer, *CIRC);
////  hst_->DL = new llvm::DataLayout(std::move(*DL));
////  hst_->Mangle = new llvm::orc::MangleAndInterner(*hst_->ES, *hst_->DL);
////  hst_->Ctx = new llvm::orc::ThreadSafeContext(std::unique_ptr<llvm::LLVMContext>(new llvm::LLVMContext()));
////  hst_->MainJD =  &hst_->ES->createJITDylib("<main>");
////  hst_->MainJD->setGenerator(llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
////                                            hst_->DL->getGlobalPrefix())));
////  llvm::cantFail(hst_->CompileLayer->add(*hst_->MainJD, llvm::orc::ThreadSafeModule(std::move(src), *hst_->Ctx)));
////  hst_->fn = (void(*)(char**, int32_t, int32_t, int32_t))(hst_->ES->lookup({hst_->MainJD}, (*hst_->Mangle)("_main"))->getAddress());



//  llvm::EngineBuilder builder(std::move(src));
//  builder.setErrorStr(&hst_->error);
//  builder.setMCJITMemoryManager(std::make_unique<llvm::SectionMemoryManager>());
//  builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
//  builder.setEngineKind(llvm::EngineKind::JIT);
//  hst_->engine = builder.create();
//  hst_->fn = (void(*)(char**, int32_t, int32_t, int32_t))(hst_->engine->getFunctionAddress("_main"));
}

std::unique_ptr<buffer> host_module::symbol(const char *name) const {
  throw std::runtime_error("not implemented");
}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */
static bool find_and_replace(std::string& str, const std::string& begin, const std::string& end, const std::string& target){
  size_t start_replace = str.find(begin);
  size_t end_replace = str.find(end, start_replace);
  if(start_replace == std::string::npos)
    return false;
  str.replace(start_replace, end_replace + 1 - start_replace, target);
  return true;
}

//static std::map<int, int> vptx = {
//  {10000, 63},
//  {10010, 64},
//  {10020, 65},
//  {11000, 70},
//  {11010, 71},
//  {11020, 72},
//  {11030, 73},
//  {11040, 73}
//};

int vptx(int version){
  if(version >= 11030) return 73;
  if(version >= 11020) return 72;
  if(version >= 11010) return 71;
  if(version >= 11000) return 70;
  if(version >= 10020) return 65;
  if(version >= 10010) return 64;
  if(version >= 10000) return 63;
  throw std::runtime_error("Triton requires CUDA 10+");
}

std::string cu_module::compile_llvm_module(llvm::Module* module, driver::device* device) {
#ifndef __HIP_PLATFORM_AMD__
  // LLVM version in use may not officially support target hardware
  int max_nvvm_cc = 75;
  int max_nvvm_ptx = 64;
#endif
  // options
  auto options = llvm::cl::getRegisteredOptions();
#ifndef __HIP_PLATFORM_AMD__
  auto* short_ptr = static_cast<llvm::cl::opt<bool>*>(options["nvptx-short-ptr"]);
  assert(short_ptr);
  short_ptr->setValue(true);
#endif

#ifndef __HIP_PLATFORM_AMD__
  // compute capability
  int cc = ((driver::cu_device*)device)->compute_capability();
  std::string sm = "sm_" + std::to_string(cc);
  // driver version
  int version;
  dispatch::cuDriverGetVersion(&version);
  int ptx = vptx(version);
  int ptx_major = ptx / 10;
  int ptx_minor = ptx % 10;
#endif
  // create
  llvm::SmallVector<char, 0> buffer;
#ifdef __HIP_PLATFORM_AMD__
  std::string rocminfo = GetAMDGPUInfo();
  
  std::string triple = "amdgcn-amd-amdhsa";
  std::string layout = "";
  std::string proc;
  std::string features;

  if (!rocminfo.empty())
  {
    proc = std::get<0>(GetFeatureStrFromGCNArchName(rocminfo));
    features = std::get<1>(GetFeatureStrFromGCNArchName(rocminfo));
  }
  else
  { // Default to MI100 params
    proc = "gfx908";
    features = "+sramecc,-xnack";
  }
  
#else
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_" + std::to_string(std::min(cc, max_nvvm_cc));
  std::string layout = "";
  std::string features = "+ptx" + std::to_string(std::min(ptx, max_nvvm_ptx));
#endif
  init_llvm();
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*module);
  // create machine
  module->setTargetTriple(triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), proc, features, opt,
                                                             llvm::Reloc::PIC_, llvm::None, llvm::CodeGenOpt::Aggressive);
  // set data layout
  if(layout.empty())
    module->setDataLayout(machine->createDataLayout());
  else
    module->setDataLayout(layout);
  // emit machine code
  for (llvm::Function &f : module->functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);

#ifdef __HIP_PLATFORM_AMD__
  // create dump files
  std::string module_name = module->getModuleIdentifier();
  std::error_code ec;

  // Save GCN ISA binary.
  std::string isabin_path = std::string("/tmp/") + module_name + std::string(".o");
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  if (ec)
  {
    std::cout << isabin_path << " was not created. error code: " << ec << std::endl;
  }
#endif

  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr, llvm::CodeGenFileType::CGFT_AssemblyFile);
#ifdef __HIP_PLATFORM_AMD__
  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr, llvm::CGFT_ObjectFile);
#endif
  pass.run(*module);

#ifdef __HIP_PLATFORM_AMD__
  // Save GCN ISA.
  std::string amdgcn_path = std::string("/tmp/") + module_name + std::string(".gcn");
  std::string result(buffer.begin(), buffer.end());
  std::ofstream amdgcn(amdgcn_path);
  amdgcn << result;
  amdgcn.close();

  // generate HASCO file
  std::string hsaco_path = std::string("/tmp/") + module_name + std::string(".hsaco");
  std::string error_message;
  int lld_result =
      llvm::sys::ExecuteAndWait("/opt/rocm/llvm/bin/ld.lld", {"/opt/rocm/llvm/bin/ld.lld", "-flavor", "gnu", "-shared", "-o", hsaco_path, isabin_path},
                                llvm::None, {}, 0, 0, &error_message);
  if (lld_result)
  {
    std::cout << "ld.lld execute fail: " << std::endl;
    std::cout << error_message << std::endl;
    std::cout << lld_result << std::endl;
  }

  return hsaco_path;
#else
  // post-process
  std::string result(buffer.begin(), buffer.end());
  find_and_replace(result, ".version", "\n", ".version " + std::to_string(ptx_major) + "." + std::to_string(ptx_minor) + "\n");
  find_and_replace(result, ".target", "\n", ".target " + sm + "\n");
  while(find_and_replace(result, "\t// begin inline asm", "\n", ""));
  while(find_and_replace(result, "\t// end inline asm", "\n", ""));
  return result;
#endif
}

void cu_module::init_from_ptx(const std::string& ptx, driver::cu_device* device) {
  // JIT compile source-code
  try{
    // use ptxas if present in PATH. Otherwise, use JIT from the driver
    std::string ptxas = "ptxas";
    std::string version;
    int use_system_ptxas = tools::exec(ptxas + " --version 2>&1", version) == 0;

    // Use PTXAS via system call
    if(use_system_ptxas){
      // compile ptx with ptxas
      char _fsrc[] = "/tmp/triton_k_XXXXXX";
      char _flog[] = "/tmp/triton_l_XXXXXX";
      mkstemp(_fsrc);
      mkstemp(_flog);
      std::string fsrc = _fsrc;
      std::string flog = _flog;
      std::ofstream ofs(fsrc);
      ofs << ptx;
      ofs.close();
      std::string cmd;
      int err;
      std::string cc = std::to_string(device->compute_capability());
      cmd = ptxas + " -v --gpu-name=sm_" + cc + " " + fsrc + " -o " + fsrc + ".o 2> " + flog;
      err = system(cmd.c_str());
      dispatch::cuModuleLoad(&*cu_, (fsrc + ".o").c_str());
      unlink(_fsrc);
      unlink(_flog);
      return;
    }

    // Use PTXAS included in driver
    CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
                          CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
                          CU_JIT_LOG_VERBOSE};
    unsigned int errbufsize = 8192;
    unsigned int logbufsize = 8192;
    char _err[errbufsize];
    char _log[logbufsize];
    void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)_err, (void*)(uintptr_t)logbufsize, (void*)_log, (void*)1};
#ifdef __HIP_PLATFORM_AMD__
    // Read HSACO.
    std::ifstream hsaco_file(ptx_, std::ios::binary | std::ios::ate);
    std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

    std::vector<unsigned char> hsaco(hsaco_file_size);
    hsaco_file.seekg(0, std::ios::beg);
    hsaco_file.read(reinterpret_cast<char*>(&hsaco[0]), hsaco_file_size);
    hsaco_file.close();
    dispatch::hipModuleLoadDataEx(&*cu_, hsaco.data(), 5, opt, optval);
#else
    dispatch::cuModuleLoadDataEx(&*cu_, ptx_.data(), 5, opt, optval);
#endif
  }
  catch(exception::cuda::invalid_ptx const &){
//#ifdef TRITON_LOG_PTX_ERROR
#ifdef __HIP_PLATFORM_AMD__
    std::cerr << "It appears that Triton produced an invalid HSACO object" << std::endl;
#else
     std::cout << ptx << std::endl;
    std::cerr << "It appears that Triton produced invalid PTX code:" << std::endl;
#endif
//    exit(1);
//#endif
    throw;
  }
}

cu_module::cu_module(driver::device* device, std::unique_ptr<llvm::Module> ll_module): module(CUmodule(), true) {
  llvm::raw_string_ostream oss(llir_);
  oss << *ll_module;
  oss.flush();
  ptx_ = compile_llvm_module(ll_module.get(), device);
  init_from_ptx(ptx_, (driver::cu_device*)device);
}

cu_module::cu_module(driver::device* device, std::string const & source) : module(CUmodule(), true), ptx_(source){
  init_from_ptx(ptx_, (driver::cu_device*)device);
}

std::unique_ptr<buffer> cu_module::symbol(const char *name) const{
  CUdeviceptr handle;
  size_t size;
  dispatch::cuModuleGetGlobal_v2(&handle, &size, *cu_, name);
  std::unique_ptr<buffer> res(new cu_buffer(size, handle, false));
  return std::move(res);
}


}
}

