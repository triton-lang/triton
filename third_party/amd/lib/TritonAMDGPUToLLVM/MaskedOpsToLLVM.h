#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_MASKEDOPSTOLLVM_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_MASKEDOPSTOLLVM_H_

#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir::triton::AMD {

void formMaskedRegions(ModuleOp module);

LogicalResult lowerMaskedOpsToLLVM(ModuleOp module,
                                   const TargetInfo &targetInfo);

Value createRegularLoadFromMaskedOp(RewriterBase &rewriter, Location loc,
                                    amdgpu::MaskedLoadOp loadOp);

LLVM::StoreOp createUnmaskedStoreFromMaskedOp(RewriterBase &rewriter,
                                              Location loc,
                                              amdgpu::MaskedStoreOp storeOp);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_MASKEDOPSTOLLVM_H_
