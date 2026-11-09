#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::nvidia_gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

void tensormap_cp_fenceproxy(Location loc, MLIRContext *ctx,
                             ConversionPatternRewriter &rewriter, Value outPtr,
                             Value inPtr) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // prepare asm operands
  auto *outAddrOpr = ptxBuilder.newAddrOperand(outPtr, "l");
  auto *inAddrOpr = ptxBuilder.newAddrOperand(inPtr, "l");
  auto *sizeOpr = ptxBuilder.newConstantOperand(TMA_SIZE_BYTES);

  // Define the instruction opcode
  auto &cp = *ptxBuilder.create("tensormap.cp_fenceproxy.global.shared::cta."
                                "tensormap::generic.release.gpu.sync.aligned");

  // Execute collectively on first warp in block
  constexpr int kWarpSize = 32;
  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));
  cp(outAddrOpr, inAddrOpr, sizeOpr).predicate(pred);

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
};

void tensormap_replace_generic(Location loc, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               NVVM::TensormapField field, Value descPtr,
                               Value newVal,
                               std::optional<int32_t> ord = std::nullopt) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (isa<LLVM::LLVMPointerType>(newVal.getType()))
    newVal = b.ptrtoint(IntegerType::get(ctx, 64), newVal);
  NVVM::TensormapReplaceOp::create(
      rewriter, loc, NVVM::TensormapFieldAttr::get(ctx, field), descPtr, newVal,
      ord ? rewriter.getI32IntegerAttr(*ord) : IntegerAttr(), Attribute());
}

template <typename Attr>
void tensormap_replace_attr(Location loc, MLIRContext *ctx,
                            ConversionPatternRewriter &rewriter,
                            NVVM::TensormapField field, Value descPtr,
                            Attr newVal) {
  // These attributes are zero in the template and the zero-fill path.
  if (static_cast<int32_t>(newVal.getValue()) == 0)
    return;
  NVVM::TensormapReplaceOp::create(rewriter, loc,
                                   NVVM::TensormapFieldAttr::get(ctx, field),
                                   descPtr, Value(), IntegerAttr(), newVal);
}

struct TensormapFenceproxyAcquireOpConversion
    : public ConvertOpToLLVMPattern<ttng::TensormapFenceproxyAcquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TensormapFenceproxyAcquireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    PTXBuilder ptxBuilder;
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // prepare asm operands
    auto *descAddrOpr = ptxBuilder.newAddrOperand(adaptor.getDescPtr(), "l");
    auto *sizeOpr = ptxBuilder.newConstantOperand(TMA_SIZE_BYTES);

    // Running the fence on one warp is faster than running it on every warp.
    constexpr int kWarpSize = 32;
    Value threadId = getThreadId(rewriter, loc);
    Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));
    auto &fence =
        *ptxBuilder.create("fence.proxy.tensormap::generic.acquire.gpu");
    fence(descAddrOpr, sizeOpr).predicate(pred);

    // Workaround for a ptxas bug missing a fence after generic.acquire.gpu.
    // TODO: remove the workaround once ptxas is fixed.
    auto &commit = *ptxBuilder.create("cp.async.bulk.commit_group");
    commit().predicate(pred);
    auto &wait = *ptxBuilder.create("cp.async.bulk.wait_group.read 0");
    wait().predicate(pred);

    ptxBuilder.launch(rewriter, loc, getVoidType());

    rewriter.eraseOp(op);
    return success();
  }
};

// cuTensorMapEncodeTiled output on SM90/SM100/SM103 for a null-address UINT8
// tensor of shape [1, 16] with a [1, 16] box, unit element strides, and default
// layout, swizzle, L2 promotion and OOB fill. Captured with libcuda 580.126.20.
constexpr int32_t tensorMapTemplate[TMA_SIZE_BYTES / sizeof(int32_t)] = {
    0, 0,          0x10, 1, 0,    0, 0,    0, 0xf, 0, 0, 0,
    0, 0x0f000000, 0,    0, 0x10, 0, 0x10, 0, 0,   0, 0, 0,
};
constexpr int templateRank = 2;

Value useLargeAllocClass(Location loc, ConversionPatternRewriter &rewriter,
                         ttng::TensormapCreateOp op,
                         ttng::TensormapCreateOp::Adaptor adaptor) {
  // Byte widths in NVVM tensormap elemtype order, up to and excluding the
  // packed types. Keep the small allocation class for packed and interleaved
  // layouts.
  constexpr int elementBytes[] = {1, 2, 4, 4, 8, 8, 2, 4, 4, 8, 2, 4, 4};
  unsigned elemType = op.getElemType();
  if (op.getInterleaveLayout() || elemType >= std::size(elementBytes))
    return {};

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value limit = b.i64_val(128 * 1024 - 1);
  Value lastByteIndex = b.i64_val(0);
  Value nonempty = b.i1_val(1);
  for (auto [i, dim] : llvm::enumerate(adaptor.getGlobalDim())) {
    nonempty = b.and_(nonempty, b.icmp_ne(dim, b.i32_val(0)));
    Value last = b.zext(i64_ty, b.sub(dim, b.i32_val(1)));
    Value stride = i == 0 ? b.i64_val(elementBytes[elemType])
                          : adaptor.getGlobalStride()[i - 1];
    // The host descriptor workaround also uses the last element's starting
    // byte. Capping each stride at the limit preserves the comparison and
    // keeps the sum of products of 32-bit dimensions from overflowing.
    lastByteIndex = b.add(lastByteIndex, b.mul(last, b.umin(stride, limit)));
  }
  return b.and_(nonempty, b.icmp_uge(lastByteIndex, limit));
}

void initialize_tma(Location loc, ConversionPatternRewriter &rewriter,
                    const NVIDIA::TargetInfo &targetInfo, Value descPtr,
                    bool useTemplate) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  constexpr int kWarpSize = 32;
  static_assert(std::size(tensorMapTemplate) == kWarpSize);
  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));

  Value fillVal = b.i32_val(0);
  if (useTemplate) {
    for (auto [i, word] : llvm::enumerate(tensorMapTemplate)) {
      if (word)
        fillVal = b.select(b.icmp_eq(threadId, b.i32_val(i)), b.i32_val(word),
                           fillVal);
    }
  }
  auto writeAddr =
      b.gep(descPtr.getType(), fillVal.getType(), descPtr, threadId);
  targetInfo.storeShared(rewriter, loc, writeAddr, fillVal, pred);
  LLVM::NVIDIA::createSyncWarp(loc, rewriter);
}

struct TensormapCreateOpConversion
    : public ConvertOpToLLVMPattern<ttng::TensormapCreateOp> {
  const NVIDIA::TargetInfo &targetInfo;

  TensormapCreateOpConversion(LLVMTypeConverter &converter,
                              const NVIDIA::TargetInfo &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::TensormapCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = getContext();

    int ptxVersion = targetInfo.getPtxVersion();
    bool needsStrideWorkaround = ptxVersion <= 85;
    auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

    int capability = targetInfo.getComputeCapability();
    bool isBlackwell = capability == 100 || capability == 103;
    bool useTemplate = capability == 90 || isBlackwell;
    initialize_tma(loc, rewriter, targetInfo, smemBase, useTemplate);
    Value isLeader = b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
    auto [previousBlock, updateBlock, continuationBlock] =
        createIfBlock(rewriter, loc, isLeader);
    (void)previousBlock;
    rewriter.setInsertionPointToStart(updateBlock);
    tensormap_replace_generic(loc, ctx, rewriter,
                              NVVM::TensormapField::GLOBAL_ADDRESS, smemBase,
                              adaptor.getGlobalAddress());
    Value useLarge;
    if (isBlackwell && ptxVersion >= 95)
      useLarge = useLargeAllocClass(loc, rewriter, op, adaptor);
    if (useLarge) {
      // PTX 9.5 sets the small allocation class when replacing the address.
      // Set the large class explicitly for tensors that meet the size cutoff.
      PTXBuilder ptxBuilder;
      auto *desc = ptxBuilder.newAddrOperand(smemBase, "r");
      auto &replace = *ptxBuilder.create(
          "tensormap.replace.tile.alloc_large.shared::cta.b1024");
      replace(desc).predicate(useLarge, "b");
      ptxBuilder.launch(rewriter, loc, void_ty(ctx));
    }
    if (op.getRank() != (useTemplate ? templateRank : 1))
      tensormap_replace_generic(loc, ctx, rewriter, NVVM::TensormapField::RANK,
                                smemBase, b.i32_val(op.getRank() - 1));
    for (auto [i, dim] : llvm::enumerate(adaptor.getBoxDim())) {
      tensormap_replace_generic(
          loc, ctx, rewriter, NVVM::TensormapField::BOX_DIM, smemBase, dim, i);
    }
    for (auto [i, dim] : llvm::enumerate(adaptor.getGlobalDim())) {
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::GLOBAL_DIM, smemBase, dim,
                                i);
    }
    for (auto [i, strideVal] : llvm::enumerate(adaptor.getGlobalStride())) {
      if (needsStrideWorkaround) {
        // Workaround for a ptxas bug
        strideVal = b.ashr(strideVal, b.i64_val(4));
      }
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::GLOBAL_STRIDE, smemBase,
                                strideVal, i);
    }
    for (auto [i, stride] : llvm::enumerate(adaptor.getElementStride())) {
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::ELEMENT_STRIDE, smemBase,
                                stride, i);
    }
    tensormap_replace_attr(
        loc, ctx, rewriter, NVVM::TensormapField::ELEMTYPE, smemBase,
        NVVM::TensormapElemtypeAttr::get(
            ctx, static_cast<NVVM::TensormapElemtype>(op.getElemType())));
    tensormap_replace_attr(
        loc, ctx, rewriter, NVVM::TensormapField::INTERLEAVE_LAYOUT, smemBase,
        NVVM::TensormapInterleaveLayoutAttr::get(
            ctx, static_cast<NVVM::TensormapInterleaveLayout>(
                     op.getInterleaveLayout())));
    tensormap_replace_attr(
        loc, ctx, rewriter, NVVM::TensormapField::SWIZZLE_MODE, smemBase,
        NVVM::TensormapSwizzleModeAttr::get(
            ctx, static_cast<NVVM::TensormapSwizzleMode>(op.getSwizzleMode())));
    tensormap_replace_attr(
        loc, ctx, rewriter, NVVM::TensormapField::FILL_MODE, smemBase,
        NVVM::TensormapFillModeAttr::get(
            ctx, static_cast<NVVM::TensormapFillMode>(op.getFillMode())));
    rewriter.setInsertionPointToStart(continuationBlock);
    // Make thread zero's field updates visible to the collective copy.
    LLVM::NVIDIA::createSyncWarp(loc, rewriter);
    tensormap_cp_fenceproxy(loc, ctx, rewriter, adaptor.getDescPtr(), smemBase);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReinterpretTensorDescOpConversion
    : public ConvertOpToLLVMPattern<ReinterpretTensorDescOp> {

  ReinterpretTensorDescOpConversion(LLVMTypeConverter &converter,
                                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(ttng::ReinterpretTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultType,
                                                       adaptor.getRawDesc());
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTMAToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<TensormapCreateOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<TensormapFenceproxyAcquireOpConversion,
               ReinterpretTensorDescOpConversion>(typeConverter, benefit);
}
