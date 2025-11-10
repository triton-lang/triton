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
    //   amdgpu::MemoryCounterWaitOp will lower s_waitcnt
    // - s_barrier syncronizes the execution for the CTA
    auto dsAttr = rewriter.getI32IntegerAttr(0);
    rewriter.create<amdgpu::MemoryCounterWaitOp>(
        op->getLoc(), /* load= */ nullptr, /* store= */ nullptr,
        /* ds= */ dsAttr);
    rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);

    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

/// Encodes the waitcnt value for AMDGPU architectures.
///
/// Note: This function duplicates the bitpacking logic from AMDGPU backend
/// (llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h), as it's not accessible from
/// llvm/include. The logic handles different encoding schemes across
/// various GPU architecture versions (pre-gfx9 to gfx11).
///
/// The waitcnt encoding uses different bit positions for each counter
/// based on the ISA version:
/// - Vmcnt (vector memory counter): tracks pending vector memory operations
/// - Expcnt (export counter): tracks pending export operations
/// - Lgkmcnt (LDS/GDS/scalar memory counter): tracks pending LDS/GDS/scalar
/// memory ops
///
/// Each architecture version has its own bit layout, Vmcnt, Expcnt and Lgkmcnt
/// are decoded as follows:
///     Vmcnt = Waitcnt[3:0]        (pre-gfx9)
///     Vmcnt = Waitcnt[15:14,3:0]  (gfx9,10)
///     Vmcnt = Waitcnt[15:10]      (gfx11)
///     Expcnt = Waitcnt[6:4]       (pre-gfx11)
///     Expcnt = Waitcnt[2:0]       (gfx11)
///     Lgkmcnt = Waitcnt[11:8]     (pre-gfx10)
///     Lgkmcnt = Waitcnt[13:8]     (gfx10)
///     Lgkmcnt = Waitcnt[9:4]      (gfx11)
static FailureOr<unsigned> encodeWaitcnt(llvm::AMDGPU::IsaVersion isaVersion,
                                         unsigned vmcnt, unsigned lgkmcnt) {
  if (isaVersion.Major == 9) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(15u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 10) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 11) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    return (vmcnt << 10) | expcnt | (lgkmcnt << 4);
  }
  return failure();
}

struct MemoryCounterWaitOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::MemoryCounterWaitOp> {
  MemoryCounterWaitOpConversion(const LLVMTypeConverter &converter,
                                const AMD::TargetInfo &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::MemoryCounterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto isaVersion = targetInfo.getIsaVersion();

    /// If major version >= fgx12, lower  to
    ///   * ROCDL::WaitDscntOp if ds is present
    ///   * ROCDL::WaitLoadcntOp if load is present
    ///   * ROCDL::WaitStorecntOp if store is present
    if (isaVersion.Major >= 12) {
      Location loc = op.getLoc();
      if (std::optional<int> ds = adaptor.getDs())
        ROCDL::WaitDscntOp::create(rewriter, loc, *ds);

      if (std::optional<int> load = adaptor.getLoad())
        ROCDL::WaitLoadcntOp::create(rewriter, loc, *load);

      if (std::optional<int> store = adaptor.getStore())
        ROCDL::WaitStorecntOp::create(rewriter, loc, *store);

      rewriter.eraseOp(op);
      return success();
    }

    /// Otherwise, lower to ROCDL::SWaitcntOp
    auto getVal = [](Attribute attr) -> unsigned {
      if (attr)
        return cast<IntegerAttr>(attr).getInt();

      // This value will be clamped to the maximum value for the target version.
      return 1024;
    };
    unsigned ds = getVal(adaptor.getDsAttr());

    unsigned vmcnt = 1024;
    Attribute load = adaptor.getLoadAttr();
    Attribute store = adaptor.getStoreAttr();
    if (load && store) {
      vmcnt = getVal(load) + getVal(store);
    } else if (load) {
      vmcnt = getVal(load);
    } else if (store) {
      vmcnt = getVal(store);
    }

    FailureOr<unsigned> waitcnt = encodeWaitcnt(isaVersion, vmcnt, ds);
    if (failed(waitcnt))
      return op.emitOpError("unsupported chipset");

    rewriter.replaceOpWithNewOp<ROCDL::SWaitcntOp>(op, *waitcnt);
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
  patterns.add<LocalBarrierOpConversion, MemoryCounterWaitOpConversion>(
      typeConverter, targetInfo, barrierBenefit);
}
