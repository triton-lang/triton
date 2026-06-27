#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

inline constexpr llvm::StringLiteral kClusterBarrierMbarOffsetAttrName =
    "ttg.mbar_offset";
inline constexpr llvm::StringLiteral kWSClusterBarrierCountAttrName =
    "ttg.ws_cluster_barrier_count";

inline void copyClusterBarrierMbarOffset(Operation *src, Operation *dst) {
  if (Attribute attr = src->getAttr(kClusterBarrierMbarOffsetAttrName))
    dst->setAttr(kClusterBarrierMbarOffsetAttrName, attr);
}

bool needsClusterBarrier(Operation *op);

void runClusterBarrierMbarAllocator(ModuleOp mod);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_
