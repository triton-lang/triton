#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
// Annotates LocalLoadOps with ttg.amdgpu.syncedByAsyncWait=true if they are
// synced by an AsyncWait.
void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod);

// Getter for the annotation applied by annotateLocalLoadsSyncedViaAsyncWait
bool isSyncedViaAsyncWait(triton::gpu::LocalLoadOp localLoadOp);

// LLVM is unable to deduce dependencies across warps and loop iterations for
// AsyncCopy and LocalLoad and will emit conservative wait counts. In triton the
// dependency is models via AsyncWait, e.g.
//   %token1 = ttg.async_copy_global_to_local/amdgpu.buffer_load_to_local
//   %token2 = ttg.async_wait %token1
//   %1      = ttg.local_load .. token %token2
// For such cases AsyncWait will emit the correct wait and the conservative
// waits are redundant and hindering performance/interleaving.
// To disable the conservative waits two alias scopes are created:
//   1) "amdgpu.AsyncCopies" will contain all AsyncCopy ops
//   2) "amdgpu.LocalLoad" will contain all LocalLoads manually synchronized via
//      AsyncWait
// ALl manually synchronized LocalLoads will additionally have "AsyncCopies" as
// a non alias scope to disable the implicit waits from the LLVM backend

// If localLoadOp has a token from an AsyncWait:
//  - Attaches "amdgpu.LocalLoad" alias scope to llLoadOp
//  - Attaches "amdgpu.AsyncCopies" as *non* alias scope to llLoadOp
void addLocalLoadNoAliasScope(triton::gpu::LocalLoadOp localLoadOp,
                              LLVM::AliasAnalysisOpInterface llLoadOp);
// Overload from above without checking the AsyncToken
void addLocalLoadNoAliasScope(LLVM::AliasAnalysisOpInterface llLoadOp);
// Attaches the "AsyncCopies" alias scope to llLoadDirectToLdsOp
void addAsyncCopyAliasScope(LLVM::AliasAnalysisOpInterface llLoadDirectToLdsOp);

} // namespace mlir::triton::AMD

#endif
