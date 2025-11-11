#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_

#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;

namespace mlir::LLVM::AMD {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

ElemLocationKey getElemCoordsFromReg(tt::LinearLayout ll, unsigned regId,
                                     MLIRContext *ctx);

llvm::MapVector<ElemLocationKey, unsigned>
mapRegToCoordinates(tt::LinearLayout ll, MLIRContext *ctx);

} // namespace mlir::LLVM::AMD
#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
