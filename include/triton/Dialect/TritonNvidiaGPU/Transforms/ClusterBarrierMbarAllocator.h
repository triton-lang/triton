#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERMBARALLOCATOR_H_

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

inline constexpr llvm::StringLiteral kClusterBarrierMbarOffsetAttrName =
    "ttg.mbar_offset";
inline constexpr llvm::StringLiteral kWSClusterBarrierCountAttrName =
    "ttg.ws_cluster_barrier_count";
inline constexpr int64_t kClusterBarrierMbarSlotSize = 16;
inline constexpr int64_t kClusterBarrierMbarBufferCount = 2;
inline constexpr int64_t kClusterBarrierMbarAllocationSize =
    kClusterBarrierMbarSlotSize * kClusterBarrierMbarBufferCount;

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
