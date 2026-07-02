#ifndef TRITON_METAL_TARGETINFO_H
#define TRITON_METAL_TARGETINFO_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::metal {

// Metal-specific target information for Apple Silicon GPUs.
// Implements SIMD group operations using Metal's simdgroup primitives,
// threadgroup barriers via metal::threadgroup_barrier(), and
// simdgroup_matrix for hardware-accelerated matrix multiply.
class MetalTargetInfo : public TargetInfoBase {
public:
  explicit MetalTargetInfo(int32_t gpuFamily) : gpuFamily(gpuFamily) {}

  // Apple Silicon SIMD width is always 32
  static constexpr int kSimdWidth = 32;
  // Threadgroup memory address space in LLVM IR
  static constexpr int kSharedAddressSpace = 3;
  // Shared memory banks (same as NVIDIA)
  static constexpr int kSharedMemoryBanks = 32;

  int getGPUFamily() const { return gpuFamily; }

  // --- TargetInfoBase interface ---

  bool supportMaximumMinimum() const override { return true; }

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  Value getGlobalTimer(RewriterBase &rewriter, Location loc) const override;

  StringRef getAtomicSyncScope(MemSyncScope scope) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               triton::gpu::AddrSpace targets) const override;

  void clusterBarrier(Location loc, RewriterBase &rewriter,
                      Operation *sourceOp) const override;

  void warpSync(Location loc, RewriterBase &rewriter) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Value val, Value pred) const override;

  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;

  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;

  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;

  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                Value selector) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op,
                  unsigned reduceLaneIdMask) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override { return kSharedAddressSpace; }

  int getAddressSpace(Attribute addressSpace) const override;

  bool supportVectorizedAtomics() const override {
    // Apple Silicon supports 32-bit atomics on device memory
    return gpuFamily >= 8; // M2+
  }

  unsigned getSharedMemoryBanks() const override { return kSharedMemoryBanks; }

  // Apple Silicon does not support ldmatrix/stmatrix PTX instructions
  bool supportLdMatrix() const override { return false; }
  bool supportStMatrix() const override { return false; }

private:
  int32_t gpuFamily;
};

} // namespace mlir::triton::metal

#endif // TRITON_METAL_TARGETINFO_H
