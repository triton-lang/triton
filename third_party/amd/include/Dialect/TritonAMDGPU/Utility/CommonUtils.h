#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp);
SmallVector<gpu::MemDescSubviewOp>
splitMemDescView(OpBuilder &builder, Location loc,
                 gpu::MemDescSubviewOp parentView,
                 SmallVector<unsigned> numSplits);
} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
