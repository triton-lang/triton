#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Tools/LayoutUtils.h"

namespace mlir {
namespace triton {
namespace NVIDIA {

// The descriptor format is described in the spec:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Unnamed fields are not used
union SMEMDescriptor {
  uint64_t descriptor;
  struct {
    uint64_t baseAddress : 14;
    uint64_t : 2;
    uint64_t leadDimensionBaseOffset : 14;
    uint64_t : 2;
    uint64_t strideDimensionBaseOffset : 14;
    uint64_t : 3;
    uint64_t matrixBaseOffset : 3;
    uint64_t : 10;
    uint64_t swizzlingMode : 2;
  };
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
  virtual MemDescOperand memLoad(int a, int b,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc) const = 0;
};

class DotOpMmaSmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaSmemLoader() = default;

  DotOpMmaSmemLoader(MMASMEMDescriptor desc, Value baseb128, LinearLayout llInv,
                     ArrayRef<unsigned> instrShape);

  static DotOpMmaSmemLoader
  build(Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
        Value smemBase, ArrayRef<unsigned> instrShape, int mmaVersion,
        bool isFp4 = false,
        std::optional<RankedTensorType> mmaTy = std::nullopt,
        std::optional<unsigned> MNdim = std::nullopt);

  static DotOpMmaSmemLoader
  build(Location loc, RewriterBase &rewriter, const LinearLayout &ll,
        int bitwidth, Value smemBase, ArrayRef<unsigned> instrShapeArray,
        int mmaVersion, std::optional<RankedTensorType> mmaTy = std::nullopt,
        std::optional<unsigned> MNdim = std::nullopt);

  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc) const;

  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc) const override;

  MMASMEMDescriptor &getDescriptor();

private:
  MMASMEMDescriptor desc;
  Value baseb128;
  LinearLayout ll;
  SmallVector<unsigned> instrShape;

  static MMASMEMDescriptor getDescriptor(const LinearLayout &ll,
                                         ArrayRef<unsigned> instrShape,
                                         int bitwidth, int mmaVersion);
};

// Helper class to load tensor memory following MMAv5 layout.
class DotOpMmaV5TmemLoader : public DotOpMmaMemLoader {
public:
  DotOpMmaV5TmemLoader() {}
  DotOpMmaV5TmemLoader(Value tensor, Value base,
                       SmallVector<unsigned int> instrShape, bool interleaved,
                       bool trans);
  MemDescOperand tmemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                          Location loc) const;

  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc) const override {
    return tmemLoad(a, b, rewriter, loc);
  }

private:
  Value base;
  bool trans;
  bool interleaved;
  bool unpacked;
  SmallVector<unsigned int> instrShape;
  int numElementsPer32b;
  int numRepM;
  int numSlicePerBlockN;
};

static Value getOffsetedBase(Value v, gpu::MemDescType memDescTy,
                             const TypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter,
                             Location loc) {
  TritonLLVMOpBuilder tb(loc, rewriter);
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, v, llvmElemTy, rewriter);
  auto offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto base = smemObj.getBase();
  return tb.gep(base.getType(), llvmElemTy, base, offset);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir
