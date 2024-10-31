#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;

// class Constant;
// class DIBuilder;
// class DbgRecord;
// class Function;
// class GlobalVariable;
// class Instruction;
// class MDNode;
// class Metadata;
// class Module;
// class Type;
// using ValueToValueMapTy = ValueMap<const Value *, WeakTrackingVH>;

namespace {

struct CloneKernelAndAugmentArgs
    : public PassInfoMixin<CloneKernelAndAugmentArgs> {
  PreservedAnalyses run(llvm::Module &module, ModuleAnalysisManager &) {
    bool modifiedCodeGen = runOnModule(module);

    return (modifiedCodeGen ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all());
  }
  bool runOnModule(llvm::Module &module);
  // isRequired being set to true keeps this pass from being skipped
  // if it has the optnone LLVM attribute
  static bool isRequired() { return true; }
};

} // end anonymous namespace

bool CloneKernelAndAugmentArgs::runOnModule(llvm::Module &module) {
  bool modifiedCodeGen = false;
  std::vector<Function *> GpuKernels;
  for (auto &function : module) {
    if (function.isIntrinsic())
      continue;
    StringRef functionName = function.getName();
    if (function.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        function.getCallingConv() == CallingConv::PTX_Kernel ||
        functionName.contains("kernel")) {
      GpuKernels.push_back(&function);
    }
  }
  for (auto &I : GpuKernels) {
    std::string AugmentedName = I->getName().str() + "Pv";
    ValueToValueMapTy VMap;
    // Add an extra ptr arg on to the instrumented/cloned kernels
    std::vector<Type *> ArgTypes;
    for (auto arg = I->arg_begin(); arg != I->arg_end(); ++arg) {
      ArgTypes.push_back(arg->getType());
    }
    ArgTypes.push_back(PointerType::get(module.getContext(), /*AddrSpace=*/0));
    FunctionType *FTy =
        FunctionType::get(I->getFunctionType()->getReturnType(), ArgTypes,
                          I->getFunctionType()->isVarArg());
    Function *NF = Function::Create(FTy, I->getLinkage(), I->getAddressSpace(),
                                    AugmentedName, &module);
    NF->copyAttributesFrom(I);
    VMap[I] = NF;

    // Get the ptr we just added to the kernel arguments
    Value *bufferPtr = &*NF->arg_end() - 1;
    Function *F = cast<Function>(VMap[I]);

    Function::arg_iterator DestI = F->arg_begin();
    for (const Argument &J : I->args()) {
      DestI->setName(J.getName());
      VMap[&J] = &*DestI++;
    }
    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, I, VMap, CloneFunctionChangeType::GlobalChanges,
                      Returns);
    modifiedCodeGen = true;
  }
  return modifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &pb) {
    pb.registerOptimizerLastEPCallback([&](ModulePassManager &mpm, auto) {
      mpm.addPass(CloneKernelAndAugmentArgs());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "clone-kernel-augment-kernel-args",
          LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
