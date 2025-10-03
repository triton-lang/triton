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
                     ArrayRef<unsigned> instrShape)
      : desc(desc), baseb128(baseb128), ll(std::move(llInv)),
        instrShape(instrShape) {}

  static DotOpMmaSmemLoader
  build(Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
        Value smemBase, ArrayRef<unsigned> instrShape, int mmaVersion,
        bool isFp4 = false,
        std::optional<RankedTensorType> mmaTy = std::nullopt,
        std::optional<unsigned> MNdim = std::nullopt) {
    auto ctx = rewriter.getContext();
    auto kOffset = str_attr("offset");
    // The handling of subviews is not as fine as it could be
    // We could compose with the identity of the memTy.getShape()
    // (at the moment llInv will be of allocShape), but then
    // we would need to handle the getReps part more carefuly
    // This way we could support more subviews that we don't
    // We can implement this generalisation in the future if needed
    auto llInv = toLinearLayout(memTy).pseudoinvert();
    auto bitwidth = memTy.getElementType().getIntOrFloatBitWidth();
    if (isFp4) {
      // hacky but well
      auto dims = to_vector(llInv.getInDimNames());
      auto trans = llInv.getBasis(dims[0], 0, kOffset) == 1;
      llInv = LinearLayout::identity1D(2, dims[trans ? 0 : 1], kOffset) * llInv;
      bitwidth /= 2;
      // The instr_shape comes in number of elements already
    }
    return build(loc, rewriter, llInv, bitwidth, smemBase, instrShape,
                 mmaVersion, mmaTy, MNdim);
  }

  static DotOpMmaSmemLoader
  build(Location loc, RewriterBase &rewriter, const LinearLayout &ll,
        int bitwidth, Value smemBase, ArrayRef<unsigned> instrShapeArray,
        int mmaVersion, std::optional<RankedTensorType> mmaTy = std::nullopt,
        std::optional<unsigned> MNdim = std::nullopt) {
    // ll is a map from two dimensions (dim0, dim1) or (row, col) into offsets
    // and blocks
    auto ctx = rewriter.getContext();
    auto kOffset = str_attr("offset");
    auto kBlock = str_attr("block");
    assert(ll.getNumOutDims() == 2);
    assert(ll.hasOutDim(kOffset) && ll.hasOutDim(kBlock));

    assert(mmaVersion == 3 || mmaVersion == 5);
    // Just needed for MMAv3
    assert(mmaTy.has_value() == (mmaVersion == 3));
    assert(MNdim.has_value() == (mmaVersion == 3));
    if (mmaVersion == 3) {
      assert(MNdim.value() < 2);
    }
    auto instrShape = to_vector(instrShapeArray);
    assert(instrShape.size() == 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Due to having a 16B alignment, we can compute the offsets in 128b
    // elements
    // TODO We should assert in the verifier that the alignment is at least 16B
    smemBase = b.ptrtoint(i32_ty, smemBase);
    Value baseSrcb128 = b.lshr(smemBase, b.i32_val(4));

    if (mmaVersion == 3) {
      auto mndim = MNdim.value();
      auto mmaLl = gpu::toLinearLayout(mmaTy.value());
      auto outDims = to_vector(mmaLl.getOutDimNames());
      auto kWarp = str_attr("warp");
      // Map from warps into the MN dimension
      auto mmaWarps = mmaLl.sublayout({kWarp}, {outDims[mndim]}) *
                      LinearLayout::identity1D(1, kWarp, outDims[1 - mndim]);
      // Map from warps to offsets in bitwidth elements
      auto warpToOffset = mmaWarps.compose(ll);
      // Map from warps to offsets in 128b elements
      auto maybeWarpToOffsetb128 =
          divideLeft(warpToOffset,
                     LinearLayout::zeros1D(1, kWarp, kOffset, 128 / bitwidth));
      assert(maybeWarpToOffsetb128.has_value());
      // zero out the first two warp bases to have a warpgroup to offset map
      auto bases = maybeWarpToOffsetb128->getBases();
      assert(maybeWarpToOffsetb128->getNumOutDims() == 2);
      bases[kWarp][0] = {0, 0};
      bases[kWarp][1] = {0, 0};
      auto warpGroupToOffsetb128 = LinearLayout(
          bases, warpToOffset.getOutDims(), /*requireSurjective=*/false);
      Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
      Value warpStrideb128 =
          applyLinearLayout(loc, rewriter, warpGroupToOffsetb128,
                            {{kWarp, warpId}})[0]
              .second;
      baseSrcb128 = b.add(baseSrcb128, warpStrideb128);
      // Increase the instruction shape to describe the size at a block level
      // as the input just describes it at a warp level
      int logwgAlongMN = 0;
      for (int i = 0; i < warpGroupToOffsetb128.getInDimSizeLog2(kWarp); i++) {
        if (warpGroupToOffsetb128.getBasis(kWarp, i, kOffset) != 0) {
          logwgAlongMN++;
        }
      }
      instrShape[mndim] *= (1 << logwgAlongMN);
    }

    for (auto [dim, instrSize] : llvm::zip(ll.getInDimNames(), instrShape)) {
      assert(instrSize <= ll.getInDimSize(dim) &&
             "Instruction shape is too large for the layout");
    }

    auto desc = getDescriptor(ll, instrShape, bitwidth, mmaVersion);

    Value baseb128 = b.zext(i64_ty, b.and_(baseSrcb128, b.i32_val(0x3FFF)));
    return {desc, baseb128, ll, instrShape};
  }

  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc) const {
    auto *ctx = loc.getContext();
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto dims = to_vector(ll.getInDimNames());
    assert((a + 1) * instrShape[0] <= ll.getInDimSize(dims[0]));
    assert((b + 1) * instrShape[1] <= ll.getInDimSize(dims[1]));
    assert(to_vector(ll.getOutDimNames()) ==
           llvm::to_vector(
               ArrayRef<StringAttr>{str_attr("offset"), str_attr("block")}));
    int32_t totalOffElems = ll.apply({{dims[0], a * instrShape[0]},
                                      {dims[1], b * instrShape[1]}})[0]
                                .second;
    int32_t smemByteOffsetb8 = totalOffElems * desc.bitwidth / 8;
    auto currDesc = desc.descriptor;
    // Take the next 0/1/2/3 bits after the 128b tile
    uint32_t mask = (desc.swizzlingByteWidth >> 4) - 1;
    currDesc.matrixBaseOffset = (smemByteOffsetb8 / 128) & mask;
    int32_t smemByteOffsetb128 = smemByteOffsetb8 >> 4;
    Value descValBase =
        tb.int_val(64, currDesc.descriptor + smemByteOffsetb128);
    // Add the base address to the descriptor
    Value descVal = tb.add(descValBase, baseb128);
    return descVal;
  }
  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc) const override {
    return {smemLoad(a, b, rewriter, loc), std::nullopt};
  }

  MMASMEMDescriptor &getDescriptor() { return desc; }

private:
  MMASMEMDescriptor desc;
  Value baseb128;
  LinearLayout ll;
  SmallVector<unsigned> instrShape;

  static MMASMEMDescriptor getDescriptor(const LinearLayout &ll,
                                         ArrayRef<unsigned> instrShape,
                                         int bitwidth, int mmaVersion) {
    // ll is a map from allocShape into offsets and blocks
    auto dims = to_vector(ll.getInDimNames());
    auto ctx = dims[0].getContext();
    auto kOffset = str_attr("offset");

    // Any CTALayout, it's not really used within getCoreMatrixLinearLayout
    auto CTALayout = triton::gpu::CTALayoutAttr::getDefault(ctx, 2);

    for (bool fp4Padded : (bitwidth == 4 ? SmallVector<bool>({false, true})
                                         : SmallVector<bool>({false}))) {
      for (auto transposed : {false, true}) {
        for (int swizzling : {0, 32, 64, 128}) {
          // FIXME: getCoreMatrixLinearLayout does not accept bitwidth < 8
          auto shmemEnc = triton::gpu::NVMMASharedEncodingAttr::get(
              ctx, swizzling, transposed, std::max(8, bitwidth), fp4Padded,
              CTALayout);
          auto shmemTile =
              getCoreMatrixLinearLayout(shmemEnc, /*disableSwizzle=*/false);
          // Rename out dims to match the original layout (in case the dims were
          // (row, col))
          auto outDims = to_vector(shmemTile.getOutDims());
          outDims[0].first = dims[0];
          outDims[1].first = dims[1];
          shmemTile = LinearLayout(shmemTile.getBases(), outDims,
                                   /*requireSurjective=*/false);
          // unpack the fp4 layout
          if (bitwidth == 4) {
            shmemTile =
                LinearLayout::identity1D(2, kOffset, dims[1]) * shmemTile;
          }

          // getCoreMatrixLinearLayout gives the k-contiguous tile
          // shmemTile is a layout onto a matrix with shape
          // If swizzling != 0: 8 x (8 * swizzling / bitwidth)
          // If swizzling == 0: 8 x (8 * 16 / bitwidth)
          assert(shmemTile.getOutDimSize(dims[0]) == 8);
          // Multiply by 2 if fp4Padded as the matrix has half the core
          // matrix has half the number of elements
          assert(shmemTile.getOutDimSize(dims[1]) * (fp4Padded ? 2 : 1) ==
                 8 * std::max(16, swizzling) / bitwidth);

          if (transposed) {
            shmemTile = transposeLinearLayout(shmemTile, {1, 0});
          }
          // Pseudoinvert as fp4 may have padding
          auto shmemTileInv = shmemTile.pseudoinvert();

          // The PTX docs are wrong in a number of ways:
          // 1) LBO can be specified for !transposed && swizzled != 0
          //    PTX says it's assumed to be 1, but  we can in fact use it
          // 2) LBO / SBO are swapped also for !transposed && swizzled == 0
          //    PTX just reports this for the transposed case
          //    EVEN MORE the computation we do here is conceptually correct
          //    and it agrees with the tensor descriptors for wgmma or
          //    tcgen05.mma but not for tcgen05.cp! tcgen05.cp follows the PTX
          //    docs!
          int lbo = 0, sbo = 0;
          int leadingDim = transposed ? 0 : 1;
          int stridedDim = transposed ? 1 : 0;
          // The lbo / sbo is defined wrt. the 128 tile, so this makes their
          // definition change for swizzling == 0 lol
          if (swizzling == 0) {
            std::swap(leadingDim, stridedDim);
          }
          auto log2RowsTile = shmemTileInv.getInDimSizeLog2(dims[leadingDim]);
          if (llvm::Log2_32(instrShape[leadingDim]) > log2RowsTile) {
            lbo = ll.getBasis(dims[leadingDim], log2RowsTile, kOffset);
          }

          auto log2ColsTile = shmemTileInv.getInDimSizeLog2(dims[stridedDim]);
          if (llvm::Log2_32(instrShape[stridedDim]) > log2ColsTile) {
            sbo = ll.getBasis(dims[stridedDim], log2ColsTile, kOffset);
          }

          // Pad the tile up to the full instruction shape with the relevant
          // stride if the instruction shape is larger than the tile
          auto bases = shmemTileInv.getBases();
          for (int d : {0, 1}) {
            // 'tile' with the atom tile according to the lbo/sbo rules
            for (int i = 1;
                 i < instrShape[d] / shmemTileInv.getInDimSize(dims[d]);
                 i *= 2) {
              auto stride = ll.getBasis(
                  dims[d], shmemTileInv.getInDimSizeLog2(dims[d]), kOffset);
              bases[dims[d]].push_back({stride * i});
            }
          }
          auto maxBasis = 0;
          for (auto dimBases : llvm::make_second_range(bases)) {
            for (auto basis : dimBases) {
              maxBasis = std::max(maxBasis, basis[0]);
            }
          }
          // Multiply by 2 or round up to the next power of 2
          shmemTileInv =
              LinearLayout(bases, {{kOffset, llvm::NextPowerOf2(maxBasis)}},
                           /*requireSurjective=*/false);
          // Add a trivial block dimension as getReps expects both layouts to
          // have the same outdims
          shmemTileInv *=
              LinearLayout::identity1D(1, dims[0], str_attr("block"));

          auto reps = getReps(ll, shmemTileInv);
          if (reps.has_value()) {
            SMEMDescriptor desc;
            desc.descriptor = mmaVersion == 5 ? 1ULL << 46 : 0ULL;
            // The lbo / sbo is defined wrt. the 128b elements
            desc.leadDimensionBaseOffset = (lbo * bitwidth / 8) >> 4;
            desc.strideDimensionBaseOffset = (sbo * bitwidth / 8) >> 4;
            switch (swizzling) {
            case 0:
              desc.swizzlingMode = 0;
              break;
            case 32:
              desc.swizzlingMode = 3;
              break;
            case 64:
              desc.swizzlingMode = 2;
              break;
            case 128:
              desc.swizzlingMode = 1;
              break;
            default:
              llvm_unreachable("Unsupported swizzling size.");
            }
            return {.descriptor = desc,
                    .swizzlingByteWidth = swizzling,
                    .bitwidth = bitwidth,
                    .transposed = transposed,
                    .fp4Padded = fp4Padded};
          }
        }
      }
    }
    llvm::report_fatal_error("Failed to find a valid layout");
  }
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
