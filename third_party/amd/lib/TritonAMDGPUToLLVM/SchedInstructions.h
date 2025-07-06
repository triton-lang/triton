#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDINSTRUCTIONS_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDINSTRUCTIONS_H_

#include "mlir/IR/Types.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// The following functions are used to collect and set side-channel information
// during to LLVM conversion/lowering to facilitate instruction scheduling
// controls.
namespace mlir::triton {
triton::DotOp getSingleDotOpIfExists(scf::ForOp forOp);
} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_SCHEDINSTRUCTIONS_H_
