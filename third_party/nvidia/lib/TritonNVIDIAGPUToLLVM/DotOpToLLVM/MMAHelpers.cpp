#include "MMAHelpers.h"

namespace mlir::triton::NVIDIA {

DotOpMmaSmemLoader::DotOpMmaSmemLoader(MMASMEMDescriptor desc, Value baseb128,
                                       LinearLayout llInv,
                                       ArrayRef<unsigned> instrShape)
    : desc(desc), baseb128(baseb128), ll(std::move(llInv)),
      instrShape(instrShape) {}

DotOpMmaSmemLoader DotOpMmaSmemLoader::build(
    Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
    Value smemBase, ArrayRef<unsigned> instrShape, int mmaVersion, bool isFp4,
    std::optional<RankedTensorType> mmaTy, std::optional<unsigned> MNdim) {
  auto ctx = rewriter.getContext();
  auto kOffset = str_attr("offset");
  auto llInv = toLinearLayout(memTy).pseudoinvert();
  auto bitwidth = memTy.getElementType().getIntOrFloatBitWidth();
  if (isFp4) {
    auto dims = to_vector(llInv.getInDimNames());
    auto trans = llInv.getBasis(dims[0], 0, kOffset) == 1;
    llInv = LinearLayout::identity1D(2, dims[trans ? 0 : 1], kOffset) * llInv;
    bitwidth /= 2;
  }
  return build(loc, rewriter, llInv, bitwidth, smemBase, instrShape, mmaVersion,
               mmaTy, MNdim);
}

DotOpMmaSmemLoader DotOpMmaSmemLoader::build(
    Location loc, RewriterBase &rewriter, const LinearLayout &ll, int bitwidth,
    Value smemBase, ArrayRef<unsigned> instrShapeArray, int mmaVersion,
    std::optional<RankedTensorType> mmaTy, std::optional<unsigned> MNdim) {
  auto ctx = rewriter.getContext();
  auto kOffset = str_attr("offset");
  auto kBlock = str_attr("block");
  assert(ll.getNumOutDims() == 2);
  assert(ll.hasOutDim(kOffset) && ll.hasOutDim(kBlock));

  assert(mmaVersion == 3 || mmaVersion == 5);
  assert(mmaTy.has_value() == (mmaVersion == 3));
  assert(MNdim.has_value() == (mmaVersion == 3));
  if (mmaVersion == 3) {
    assert(MNdim.value() < 2);
  }
  auto instrShape = to_vector(instrShapeArray);
  assert(instrShape.size() == 2);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  smemBase = b.ptrtoint(i32_ty, smemBase);
  Value baseSrcb128 = b.lshr(smemBase, b.i32_val(4));

  if (mmaVersion == 3) {
    auto mndim = MNdim.value();
    auto mmaLl = gpu::toLinearLayout(mmaTy.value());
    auto outDims = to_vector(mmaLl.getOutDimNames());
    auto kWarp = str_attr("warp");
    auto mmaWarps = mmaLl.sublayout({kWarp}, {outDims[mndim]}) *
                    LinearLayout::identity1D(1, kWarp, outDims[1 - mndim]);
    auto warpToOffset = mmaWarps.compose(ll);
    auto maybeWarpToOffsetb128 = divideLeft(
        warpToOffset, LinearLayout::zeros1D(1, kWarp, kOffset, 128 / bitwidth));
    assert(maybeWarpToOffsetb128.has_value());
    auto bases = maybeWarpToOffsetb128->getBases();
    assert(maybeWarpToOffsetb128->getNumOutDims() == 2);
    bases[kWarp][0] = {0, 0};
    bases[kWarp][1] = {0, 0};
    auto warpGroupToOffsetb128 = LinearLayout(bases, warpToOffset.getOutDims(),
                                              /*requireSurjective=*/false);
    Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
    Value warpStrideb128 =
        applyLinearLayout(loc, rewriter, warpGroupToOffsetb128,
                          {{kWarp, warpId}})[0]
            .second;
    baseSrcb128 = b.add(baseSrcb128, warpStrideb128);
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

Value DotOpMmaSmemLoader::smemLoad(int a, int b,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc) const {
  [[maybe_unused]] auto *ctx = loc.getContext();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto dims = to_vector(ll.getInDimNames());
  assert((a + 1) * instrShape[0] <= ll.getInDimSize(dims[0]));
  assert((b + 1) * instrShape[1] <= ll.getInDimSize(dims[1]));
  assert(to_vector(ll.getOutDimNames()) ==
         llvm::to_vector(
             ArrayRef<StringAttr>{str_attr("offset"), str_attr("block")}));
  int32_t totalOffElems =
      ll.apply({{dims[0], a * instrShape[0]}, {dims[1], b * instrShape[1]}})[0]
          .second;
  int32_t smemByteOffsetb8 = totalOffElems * desc.bitwidth / 8;
  auto currDesc = desc.descriptor;
  uint32_t mask = (desc.swizzlingByteWidth >> 4) - 1;
  currDesc.matrixBaseOffset = (smemByteOffsetb8 / 128) & mask;
  int32_t smemByteOffsetb128 = smemByteOffsetb8 >> 4;
  Value descValBase = tb.int_val(64, currDesc.descriptor + smemByteOffsetb128);
  Value descVal = tb.add(descValBase, baseb128);
  return descVal;
}

MemDescOperand DotOpMmaSmemLoader::memLoad(int a, int b,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc) const {
  return {smemLoad(a, b, rewriter, loc), std::nullopt};
}

MMASMEMDescriptor &DotOpMmaSmemLoader::getDescriptor() { return desc; }

MMASMEMDescriptor
DotOpMmaSmemLoader::getDescriptor(const LinearLayout &ll,
                                  ArrayRef<unsigned> instrShape, int bitwidth,
                                  int mmaVersion) {
  auto dims = to_vector(ll.getInDimNames());
  auto ctx = dims[0].getContext();
  auto kOffset = str_attr("offset");

  auto CTALayout = triton::gpu::CTALayoutAttr::getDefault(ctx, 2);

  for (bool fp4Padded : (bitwidth == 4 ? SmallVector<bool>({false, true})
                                       : SmallVector<bool>({false}))) {
    for (auto transposed : {false, true}) {
      for (int swizzling : {0, 32, 64, 128}) {
        auto shmemEnc = triton::gpu::NVMMASharedEncodingAttr::get(
            ctx, swizzling, transposed, std::max(8, bitwidth), fp4Padded,
            CTALayout);
        auto shmemTile =
            getCoreMatrixLinearLayout(shmemEnc, /*disableSwizzle=*/false);
        auto outDims = to_vector(shmemTile.getOutDims());
        outDims[0].first = dims[0];
        outDims[1].first = dims[1];
        shmemTile = LinearLayout(shmemTile.getBases(), outDims,
                                 /*requireSurjective=*/false);
        if (bitwidth == 4) {
          shmemTile = LinearLayout::identity1D(2, kOffset, dims[1]) * shmemTile;
        }

        if (transposed) {
          shmemTile = transposeLinearLayout(shmemTile, {1, 0});
        }
        auto shmemTileInv = shmemTile.pseudoinvert();

        int lbo = 0, sbo = 0;
        int leadingDim = transposed ? 0 : 1;
        int stridedDim = transposed ? 1 : 0;
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

        auto bases = shmemTileInv.getBases();
        for (int d : {0, 1}) {
          for (int i = 1;
               i < instrShape[d] / shmemTileInv.getInDimSize(dims[d]); i *= 2) {
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
        shmemTileInv =
            LinearLayout(bases, {{kOffset, llvm::NextPowerOf2(maxBasis)}},
                         /*requireSurjective=*/false);
        shmemTileInv *= LinearLayout::identity1D(1, dims[0], str_attr("block"));

        auto reps = getReps(ll, shmemTileInv);
        if (reps.has_value()) {
          SMEMDescriptor desc;
          desc.descriptor = mmaVersion == 5 ? 1ULL << 46 : 0ULL;
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

} // namespace mlir::triton::NVIDIA
