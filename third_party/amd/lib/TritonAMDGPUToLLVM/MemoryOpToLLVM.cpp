#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using ::mlir::triton::gpu::MemDescType;

namespace {
template <typename LocalLoadOpType>
class TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<LocalLoadOpType> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<LocalLoadOpType>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename LocalLoadOpType::Adaptor;

  static constexpr bool isPackedLoad =
      std::is_same_v<triton::amdgpu::LocalLoadPackedTransposedOp,
                     LocalLoadOpType>;

  LogicalResult
  matchAndRewrite(LocalLoadOpType op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    // 64 is the number of bytes ds_read_tr
    unsigned needContigReg = 64 / bitwidth;
    // 16 is the number of lanes that participate in the data shuffle
    unsigned needContigLane = 16;

    if (!canUseTransLoad(op, srcTy, dstTy, bitwidth, needContigReg,
                         needContigLane)) {
      return failure();
    }
    auto shape = srcTy.getShape();
    // FP4 are packed into i8 so the real bitwidth is different
    auto llBitwidth = isPackedLoad ? 4 : llvmElemTy.getIntOrFloatBitWidth();
    auto ldsTransLayout = triton::gpu::chooseDsReadB64TrLayout(
        dstTy.getEncoding(), shape, llBitwidth);

    int vecSize = getLayoutVectorization(srcTy, ldsTransLayout);
    if (vecSize < needContigReg)
      return failure();

    return lowerSharedToDotOperandTransLL(op, ldsTransLayout, adaptor,
                                          typeConverter, rewriter);
  }

private:
  bool checkLayoutProperties(MemDescType srcTy, RankedTensorType dstTy,
                             unsigned needContigReg,
                             unsigned needContigLane) const {
    auto srcOrder = triton::gpu::getOrder(srcTy);
    auto dstOrder = triton::gpu::getOrder(dstTy);

    // Check that the contiguity of srcTy and dstTy don't match
    // this is because ds_read_tr will reshuffle the data to
    // the opposite contiguity
    if (dstOrder[0] == srcOrder[0])
      return false;

    auto dstLL = triton::gpu::toLinearLayout(dstTy);
    SmallVector<StringAttr> outDimNames(dstLL.getOutDimNames());
    std::swap(outDimNames[0], outDimNames[1]);
    auto dstTrLL = dstLL.transposeOuts(outDimNames);

    // Check the main requirements for the ds_read_tr instruction: contiguity
    // of reg/lane. This is because ds_read_tr works on a block of 16 lanes
    // with each holding 64 bits of data. Each lane will load 64 bits of
    // contiguous data and then share it among the lane dimension.
    // This means that there needs to be a check that each lane owns
    // 64 bit of contig data and that the communicating lanes are contiguous.
    // In order to do this, we use ll.getNumConsecutiveInOut() which
    // can get the contiguity of the first component of the first
    // dimension.
    // Since the data might be dim0 or dim1 contiguous we need both the
    // dstLL and the dstTrLL: one to check the register dimension
    // contiguity and the other to check the lane dimension one.
    bool dim1Contig = dstOrder[0] == 1;
    auto dstLLDim0Contig = dim1Contig ? dstTrLL : dstLL;
    auto dstLLDim1Contig = dim1Contig ? dstLL : dstTrLL;
    int contigRegisters = dstLLDim0Contig.getNumConsecutiveInOut();

    assert(dstLLDim0Contig.getBases().begin()->first == "register");
    SmallVector<StringAttr> subLayoutInDims(
        llvm::drop_begin(dstLLDim0Contig.getInDimNames()));
    SmallVector<StringAttr> subLayoutOutDims(dstLLDim0Contig.getOutDimNames());
    auto dstLLOnlyLaneWarp =
        dstLLDim1Contig.sublayout(subLayoutInDims, subLayoutOutDims);
    int contigLanes = dstLLOnlyLaneWarp.getNumConsecutiveInOut();

    // Check that the tile size used by ds_read_tr (KxM/N = 4x16 for 16-bit
    // elements) is contiguous both in terms of registers dimension and in
    // terms of lane dimension. If that is the case then we can use ds_read_tr
    return contigRegisters >= needContigReg && contigLanes >= needContigLane;
  }

  bool checkCurrentLimitation(unsigned bitwidth) const {
    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitwidth != 16 && bitwidth != 8) {
      return false;
    }

    return true;
  }

  bool canUseTransLoad(Operation *localLoad, MemDescType srcTy,
                       RankedTensorType dstTy, unsigned bitwidth,
                       unsigned needContigReg, unsigned needContigLane) const {
    // Packed loads need to always map to ds_read_tr
    if constexpr (isPackedLoad) {
      return true;
    }

    if (!targetInfo.canUseLDSTransLoad(bitwidth)) {
      return false;
    }

    if (!checkCurrentLimitation(bitwidth)) {
      return false;
    }

    if (!checkLayoutProperties(srcTy, dstTy, needContigReg, needContigLane)) {
      return false;
    }

    return true;
  }

  int getLayoutVectorization(MemDescType srcTy,
                             LinearLayout ldsTransLayout) const {
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout regToSharedLayout = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      regToSharedLayout = ldsTransLayout.invertAndCompose(sharedLL);
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      regToSharedLayout = ldsTransLayout.invertAndCompose(sharedLL);
    }

    // Determine how many consecutive registers map to consecutive shmem
    // elements in out-dimension offsetN.
    int vecElems = regToSharedLayout.getNumConsecutiveInOut();
    if (paddedEnc) {
      vecElems = std::min(vecElems, int(paddedEnc.getMinInterval()));
    }

    return vecElems;
  }

  LogicalResult
  lowerSharedToDotOperandTransLL(LocalLoadOpType op,
                                 LinearLayout ldsTransLayout, OpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> outVals;
    SmallVector<Value> elemsI32;
    mlir::Type retTy = dstTy;
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    bool valid = emitTransferBetweenRegistersAndShared(
        ldsTransLayout, srcTy, llvmElemTy,
        /*maxVecElems=*/std::nullopt, smemObj, loc, rewriter, targetInfo,
        laneId, warpId, [&](VectorType vecTy, Value vecAddr) {
          if constexpr (isPackedLoad) {
            assert(bitwidth == 8);
            auto numElems = vecTy.getNumElements();
            auto numElemsI32 = (numElems * bitwidth / 32);
            auto i32VecTy = VectorType::get(numElemsI32, i32_ty);
            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr4_b64>(loc, i32VecTy, vecAddr);
            auto res = b.bitcast(dsReadOp.getResult(), vecTy);
            Value vecVal = res.getResult();
            for (int v = 0; v < vecTy.getNumElements(); v++) {
              outVals.push_back(
                  b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
            }
          } else if (bitwidth == 16) {
            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr16_b64>(loc, vecTy, vecAddr);
            if constexpr (!isPackedLoad) {
              AMD::addLocalLoadNoAliasScope(op, dsReadOp);
            }
            Value vecVal = dsReadOp.getResult();
            for (int v = 0; v < vecTy.getNumElements(); v++) {
              outVals.push_back(
                  b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
            }
          } else {
            // pack elements in i32 vectors
            auto numElems = vecTy.getNumElements();
            auto numElemsI32 = (numElems * bitwidth / 32);
            auto i32VecTy = VectorType::get(numElemsI32, i32_ty);

            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr8_b64>(loc, i32VecTy, vecAddr);
            if constexpr (!isPackedLoad) {
              AMD::addLocalLoadNoAliasScope(op, dsReadOp);
            }
            Value vecVal = dsReadOp.getResult();
            for (auto i = 0; i < numElemsI32; ++i) {
              elemsI32.push_back(
                  b.extract_element(i32_ty, vecVal, b.i32_val(i)));
            }
          }
        });

    // unpack i32 vectors and cast to native type
    if (bitwidth != 16) {
      auto numElemsPerVec = 32 / bitwidth;
      auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
      for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
        auto vec = b.bitcast(elemsI32[v], vecTy);
        for (int i = 0; i < numElemsPerVec; ++i)
          outVals.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
      }

      retTy = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(outVals.size(), llvmElemTy));
    }
    assert(valid && "Failed to emit LDS transpose load operations");
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, retTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class LocalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp> {
public:
  LocalBarrierOpConversion(const LLVMTypeConverter &converter,
                           const AMD::TargetInfo &targetInfo,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::LocalBarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCDNA(targetInfo.getISAFamily()))
      return failure();
    // In CDNA we can lower local_barrier to s_waitcnt + s_barrier
    // - s_waitcnt specifies how many operations to VMEM/LDS can be outstanding
    //   when the instruction completes.
    //   In this case we require 0 outstanding LDS operations
    // - s_barrier syncronizes the execution for the CTA
    constexpr int32_t ldsOnlyBits = ~(0x1f << 8);
    Location loc = op->getLoc();
    ROCDL::SWaitcntOp::create(rewriter, loc, ldsOnlyBits);
    rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);

    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<TransLocalLoadOpConversion<triton::gpu::LocalLoadOp>>(
      typeConverter, targetInfo, transBenefit);
  patterns.add<
      TransLocalLoadOpConversion<triton::amdgpu::LocalLoadPackedTransposedOp>>(
      typeConverter, targetInfo, benefit);
  patterns.add<LocalBarrierOpConversion>(typeConverter, targetInfo,
                                         barrierBenefit);
}
