#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::LLVM::AMD {

// Determine the order in which CTA tiles are laid out across the tensor.
// That is, create vector of dimensions from fastest to slowest varying.
SmallVector<unsigned> getCTATileOrder(MLIRContext *ctx,
                                      const mlir::triton::LinearLayout &layout);

// Infer LinearLayout from the amd::extract_slice
mlir::triton::LinearLayout
inferLinearLayoutFromExtractSlice(MLIRContext *ctx,
                                  RankedTensorType tensorType);

// Infer LinearLayout from the amd::concat
mlir::triton::LinearLayout
inferLinearLayoutFromConcat(MLIRContext *ctx, RankedTensorType tensorType);

} // namespace mlir::LLVM::AMD
#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
