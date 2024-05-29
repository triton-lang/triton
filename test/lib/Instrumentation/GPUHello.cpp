#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;

namespace {

struct GPUHello : public PassInfoMixin<GPUHello> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool ModifiedCodeGen = runOnModule(M);

    return (ModifiedCodeGen ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all());
  }
  bool runOnModule(llvm::Module &M);
  // isRequired being set to true keeps this pass from being skipped
  // if it has the optnone LLVM attribute
  static bool isRequired() { return true; }
};

} // end anonymous namespace

bool GPUHello::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;

  for (auto &F : M) {
    if (F.isIntrinsic())
      continue;
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        F.getCallingConv() == CallingConv::PTX_Kernel) {
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();
          std::string SourceInfo =
              (F.getName() + "\t" + DL->getFilename() + ":" +
               Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                  .str();

          errs() << "Hello From First Instruction of GPU Kernel: " << SourceInfo
                 << "\n";
          return ModifiedCodeGen;
        }
      }
    }
  }
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
      MPM.addPass(GPUHello());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "gpu-hello", LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
