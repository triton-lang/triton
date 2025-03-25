//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for LLVMXPU dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the LLVMXPU dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerLLVMXPUDialectTranslation(DialectRegistry &registry);

/// Register the LLVMXPU dialect and the translation from it in the registry
/// associated with the given context.
void registerLLVMXPUDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_H
