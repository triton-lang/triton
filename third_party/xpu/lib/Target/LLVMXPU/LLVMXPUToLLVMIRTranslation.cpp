//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===- LLVMXPUToLLVMIRTranslation.cpp - Translate LLVMXPU to LLVM IR
//------------===//
//
// This file implements a translation between the MLIR LLVMXPU dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsXPU.h" //llvm::Intrinsic
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Target/LLVMXPU/LLVMXPUToLLVMIRTranslation.h"
// clang-format on

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMXPU dialect to LLVM IR.
class LLVMXPUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "triton/Dialect/LLVMXPU/IR/LLVMXPUConversions.inc"
    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    auto generateMetadata = [&](int dim, StringRef name) {
      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmContext, name),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmContext), dim))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmContext, llvmMetadata);
      moduleTranslation.getOrInsertNamedModuleMetadata("xpu.annotations")
          ->addOperand(llvmMetadataNode);
    };

    return success();
  }
};
} // namespace

void mlir::registerLLVMXPUDialectTranslation(DialectRegistry &registry) {
  registry.insert<XPU::LLVMXPUDialect>();
  registry.addExtension(+[](MLIRContext *ctx, XPU::LLVMXPUDialect *dialect) {
    dialect->addInterfaces<LLVMXPUDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerLLVMXPUDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMXPUDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
