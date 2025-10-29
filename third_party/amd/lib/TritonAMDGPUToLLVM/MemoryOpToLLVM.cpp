#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
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
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitWidth != 16 && bitWidth != 8) {
      return failure();
    }
    // FP4 packed along M/N are not supported yet on GFX1250
    if (targetInfo.getISAFamily() == AMD::ISAFamily::GFX1250 && isPackedLoad) {
      return failure();
    }

    return lowerSharedToDotOperandTransLL(op, adaptor, typeConverter, rewriter);
  }

private:
  LogicalResult
  lowerSharedToDotOperandTransLL(LocalLoadOpType op, OpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    mlir::Type retTy = dstTy;
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto calcPaddedOffset = [&](Value smemOffset) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();
      if (auto paddedLayout = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
              srcTy.getEncoding())) {
        // Apply the offset needed for padding.
        Value padOffset = emitPadding(loc, rewriter, paddedLayout, bitWidth,
                                      smemOffset, /*offsetInBytes=*/true);
        smemOffset = b.add(smemOffset, padOffset);
      }
      return smemOffset;
    };

    auto shape = srcTy.getShape();
    auto ldsTransLoadParams = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (!ldsTransLoadParams)
      return failure();
    // FP4 are packed into i8 so the real bitWidth is different
    auto llBitWidth = isPackedLoad ? 4 : llvmElemTy.getIntOrFloatBitWidth();
    auto ldsTransLayout = triton::gpu::chooseDsReadTrLayout(
        dstTy.getEncoding(), shape, llBitWidth,
        ldsTransLoadParams->instBitWidth,
        ldsTransLoadParams->numLanesInShuffleGroup);

    // Check that we have computed a layout
    if (!ldsTransLayout) {
      return failure();
    }

    auto smemPtrTy = ptr_ty(ctx, 3);
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = ldsTransLayout->invertAndCompose(sharedLL);
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      cvt = ldsTransLayout->invertAndCompose(sharedLL);
    }
    // Check that we will be able to vectorize the load.
    // Need to have exactly needContigReg, otherwise we can't use ds_read_tr
    auto [elemsPerVec, permutation] = largestVectorisation(
        ctx, cvt, bitWidth, ldsTransLoadParams->needContigReg);

    if (paddedEnc)
      elemsPerVec = std::min<int>(elemsPerVec, paddedEnc.getMinInterval());

    if (elemsPerVec != ldsTransLoadParams->needContigReg)
      return failure();

    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
    auto lowerInst = [&](RewriterBase &rewriter, Location loc,
                         ArrayRef<Value> inVals, Value vecAddr, int idx,
                         VectorType vTy) -> SmallVector<Value> {
      assert(bitWidth == 16 || bitWidth == 8);
      Value dsReadTr;
      // tr16 instructions return vectors of bf16/f16 while "tr8" instructions
      // return vectors of i32. Generate the corresponding i32 vector
      auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
      auto vTyI32 = VectorType::get(numElemsI32, i32_ty);
      switch (targetInfo.getISAFamily()) {
      case AMD::ISAFamily::GFX1250: {
        if (bitWidth == 16) {
          dsReadTr = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, "llvm.amdgcn.ds.load.tr16.b128", {vTy},
                         {vecAddr})
                         .getResult(0);
        } else
          dsReadTr = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, "llvm.amdgcn.ds.load.tr8.b64", {vTyI32},
                         {vecAddr})
                         .getResult(0);
        break;
      }
      case AMD::ISAFamily::CDNA4: {
        if (bitWidth == 16) {
          dsReadTr =
              ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
        } else {
          if (isPackedLoad) {
            dsReadTr =
                ROCDL::ds_read_tr4_b64::create(rewriter, loc, vTyI32, vecAddr);
          } else {
            dsReadTr =
                ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
          }
        }
        break;
      }
      default:
        return {};
      }
      // GFX1250 is currently using LLVM intrinsics so it cannot cast it to
      // AliasAnalysisOpInterface
      if (targetInfo.getISAFamily() != AMD::ISAFamily::GFX1250)
        AMD::addLocalLoadNoAliasScope(
            op, cast<LLVM::AliasAnalysisOpInterface>(dsReadTr.getDefiningOp()));
      Value vecVal = b.bitcast(dsReadTr, vTy);
      SmallVector<Value> loadedVals;
      for (int v = 0; v < vTy.getNumElements(); v++) {
        loadedVals.push_back(
            b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
      }

      return loadedVals;
    };

    SmallVector<Value> outVals = lowerLdSt(
        loc, rewriter.getContext(), cvt, {}, // Input for store, output for load
        llvmElemTy, smemObj.getBase(), calcPaddedOffset, affineOffset,
        maskSpanAffineOffset, laneId, warpId, rewriter, targetInfo,
        ldsTransLoadParams->needContigReg, lowerInst);
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
