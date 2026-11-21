#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_MMA_HELPERS_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_MMA_HELPERS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {
class ConversionPatternRewriter;
class RewriterBase;
class TypeConverter;

namespace triton {
namespace gpu {
class MemDescType;
} // namespace gpu

namespace NVIDIA {

// The descriptor format is described in the spec:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Reserved fields are named so bit_cast preserves every descriptor bit.
struct SMEMDescriptor {
  uint64_t baseAddress : 14;
  uint64_t reserved0 : 2;
  uint64_t leadDimensionBaseOffset : 14;
  uint64_t reserved1 : 2;
  uint64_t strideDimensionBaseOffset : 14;
  uint64_t reserved2 : 3;
  uint64_t matrixBaseOffset : 3;
  uint64_t leadDimensionAbsoluteMode : 1;
  uint64_t reserved3 : 9;
  uint64_t swizzlingMode : 2;
};

struct MMASMEMDescriptor {
  SMEMDescriptor descriptor;
  int32_t swizzlingByteWidth;
  int32_t bitwidth;
  bool transposed;
  bool fp4Padded;
};

struct MemDescOperand {
  Value base;
  std::optional<int> offset;
};

// Abstract class to calculate the address of a shared or tensor memory slice.
class DotOpMmaMemLoader {
public:
  virtual ~DotOpMmaMemLoader() = default;
  // Given the starting coordinates of the logical tensor (i.e. reps *
  // ctaTileSize), return the associated memory descriptor for SMEM / TMEM.
  virtual MemDescOperand memLoad(int a, int b,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc, int kSize = 0) const = 0;
};

class DotOpMmaSmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaSmemLoader() = default;

  DotOpMmaSmemLoader(MMASMEMDescriptor desc, Value baseb128, LinearLayout llInv)
      : desc(desc), baseb128(baseb128), ll(std::move(llInv)) {}

  static FailureOr<DotOpMmaSmemLoader>
  build(Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
        Value smemBase, ArrayRef<unsigned> instrShape, unsigned MNdim,
        int mmaVersion, bool isFp4 = false,
        std::optional<RankedTensorType> mmaTy = std::nullopt);

  static FailureOr<DotOpMmaSmemLoader>
  build(Location loc, RewriterBase &rewriter, const LinearLayout &ll,
        int bitwidth, Value smemBase, ArrayRef<unsigned> instrShape,
        unsigned MNdim, int mmaVersion,
        std::optional<RankedTensorType> mmaTy = std::nullopt);

  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc, int kSize = 0) const;
  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc, int kSize = 0) const override;

  MMASMEMDescriptor &getDescriptor() { return desc; }

private:
  MMASMEMDescriptor desc;
  Value baseb128;
  LinearLayout ll;

  static FailureOr<MMASMEMDescriptor>
  getDescriptor(Location loc, const LinearLayout &ll,
                ArrayRef<unsigned> instrShape, int bitwidth, unsigned MNdim,
                int mmaVersion);
};

Value getOffsetedBase(Value v, gpu::MemDescType memDescTy,
                      const TypeConverter *typeConverter,
                      ConversionPatternRewriter &rewriter, Location loc);

} // namespace NVIDIA
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_MMA_HELPERS_H
