#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_

#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;

namespace mlir::LLVM::AMD {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

ElemLocationKey getElemCoordinatesFromRegisters(tt::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx);

std::optional<int> getRegFromCoordinates(tt::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx);

} // namespace mlir::LLVM::AMD
#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
