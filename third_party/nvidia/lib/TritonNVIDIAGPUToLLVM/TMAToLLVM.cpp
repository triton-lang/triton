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

#include <algorithm>
#include <array>
#include <bitset>

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
  // The zero-initialized descriptor already has the default attribute values.
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

using TensormapWords = std::array<uint32_t, TMA_SIZE_BYTES / sizeof(uint32_t)>;
constexpr unsigned maxTensorMapRank = 5;
constexpr uint64_t largeAllocLastByte = 128 * 1024 - 1;
constexpr uint32_t largeAllocBit = 1u << 21;

unsigned getAllocClassElementBytes(ttng::TensormapCreateOp op) {
  // Byte widths in NVVM tensormap elemtype order, up to and excluding the
  // packed types. Keep the small allocation class for packed and interleaved
  // layouts.
  constexpr int elementBytes[] = {1, 2, 4, 4, 8, 8, 2, 4, 4, 8, 2, 4, 4};
  unsigned elemType = op.getElemType();
  if (op.getInterleaveLayout() || elemType >= std::size(elementBytes))
    return 0;
  return elementBytes[elemType];
}

std::optional<uint64_t> getKnownInt(ModuleAxisInfoAnalysis &axisInfo,
                                    Value value) {
  if (const AxisInfo *info = axisInfo.getAxisInfo(value))
    if (const auto &constant = info->getConstantValue())
      return constant->getZExtValue();
  return std::nullopt;
}

struct TensormapTemplate {
  TensormapWords words{};
  std::bitset<maxTensorMapRank> boxDims, globalDims, globalStrides,
      elementStrides;
  std::optional<bool> largeAlloc;
};

// Pack the fields written by tensormap.replace on SM90/SM100/SM103. The
// offsets and encodings were measured by replacing each field independently
// and comparing the resulting words with cuTensorMapEncodeTiled output.
// Unknown values are patched in shared memory. The trailing CUDA metadata is
// zero because Triton sets the transaction size and shared-memory layout
// separately.
std::optional<TensormapTemplate>
packTensormapTemplate(ttng::TensormapCreateOp op,
                      ModuleAxisInfoAnalysis &axisInfo, int capability,
                      int ptxVersion) {
  unsigned rank = op.getRank();
  unsigned elemType = op.getElemType();
  unsigned interleave = op.getInterleaveLayout();
  unsigned swizzle = op.getSwizzleMode();
  unsigned fill = op.getFillMode();
  // In PTX enum order, including the alternate encoding of TF32.
  constexpr uint32_t tf32Bit = 1u << 16;
  constexpr uint32_t elemTypeBits[] = {
      0,
      1 << 7,
      2 << 7,
      3 << 7,
      4 << 7,
      5 << 7,
      6 << 7,
      7 << 7,
      8 << 7,
      9 << 7,
      10 << 7,
      tf32Bit | (7 << 7),
      tf32Bit | (8 << 7),
      11 << 7,
      12 << 7,
      13 << 7,
  };
  if (!rank || rank > maxTensorMapRank || elemType >= std::size(elemTypeBits) ||
      interleave > 2 || swizzle > 4 || fill > 1 ||
      (elemType >= 13 && (capability == 90 || ptxVersion < 87)) ||
      (swizzle == 4 && (capability != 103 || ptxVersion < 88)))
    return std::nullopt;

  TensormapTemplate result;
  auto &words = result.words;
  words[2] = ((rank - 1) << 4) | elemTypeBits[elemType] | (interleave << 11) |
             (swizzle == 4 ? 1u << 22 : swizzle << 13) | (fill << 15);

  for (auto [i, operand] : llvm::enumerate(op.getBoxDim())) {
    if (auto dim = getKnownInt(axisInfo, operand); dim && *dim && *dim <= 256) {
      // Dimension 0 is in the high byte of word 13. The other four occupy
      // the bytes of word 14.
      unsigned word = i == 0 ? 13 : 14;
      unsigned shift = i == 0 ? 24 : 8 * (i - 1);
      words[word] |= static_cast<uint32_t>(*dim - 1) << shift;
      result.boxDims.set(i);
    }
  }

  std::array<uint64_t, maxTensorMapRank> globalDims{}, strideBytes{};
  for (auto [i, operand] : llvm::enumerate(op.getGlobalDim())) {
    if (auto dim = getKnownInt(axisInfo, operand)) {
      globalDims[i] = *dim;
      words[8 + i] = static_cast<uint32_t>(*dim) - 1;
      result.globalDims.set(i);
    }
  }
  for (auto [i, operand] : llvm::enumerate(op.getGlobalStride())) {
    if (auto stride = getKnownInt(axisInfo, operand);
        stride && *stride % 16 == 0 && *stride < (1ull << 40)) {
      strideBytes[i + 1] = *stride;
      // 36 bits in units of 16 bytes, with the high nibbles in word 7.
      uint64_t encoded = *stride >> 4;
      words[3 + i] = static_cast<uint32_t>(encoded);
      words[7] |= static_cast<uint32_t>(encoded >> 32) << (4 * i);
      result.globalStrides.set(i);
    }
  }
  for (auto [i, operand] : llvm::enumerate(op.getElementStride())) {
    if (auto stride = getKnownInt(axisInfo, operand);
        stride && *stride && *stride <= 8) {
      words[13] |= static_cast<uint32_t>(*stride - 1) << (3 * i);
      result.elementStrides.set(i);
    }
  }

  strideBytes[0] = getAllocClassElementBytes(op);
  if (ptxVersion >= 95 && capability != 90 && strideBytes[0] &&
      result.globalDims.count() == rank &&
      result.globalStrides.count() == rank - 1) {
    uint64_t lastByte = 0;
    bool nonempty = true;
    for (unsigned i = 0; i < rank; ++i) {
      nonempty &= globalDims[i] != 0;
      if (globalDims[i])
        lastByte +=
            (globalDims[i] - 1) * std::min(strideBytes[i], largeAllocLastByte);
    }
    result.largeAlloc = nonempty && lastByte >= largeAllocLastByte;
    if (*result.largeAlloc)
      words[2] |= largeAllocBit;
  }
  return result;
}

Value useLargeAllocClass(Location loc, ConversionPatternRewriter &rewriter,
                         ttng::TensormapCreateOp op,
                         ttng::TensormapCreateOp::Adaptor adaptor) {
  unsigned elementBytes = getAllocClassElementBytes(op);
  if (!elementBytes)
    return {};
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value limit = b.i64_val(largeAllocLastByte);
  Value lastByteIndex = b.i64_val(0);
  Value nonempty = b.i1_val(1);
  for (auto [i, dim] : llvm::enumerate(adaptor.getGlobalDim())) {
    nonempty = b.and_(nonempty, b.icmp_ne(dim, b.i32_val(0)));
    Value last = b.zext(i64_ty, b.sub(dim, b.i32_val(1)));
    Value stride =
        i == 0 ? b.i64_val(elementBytes) : adaptor.getGlobalStride()[i - 1];
    // The host descriptor workaround also uses the last element's starting
    // byte. Capping each stride at the limit preserves the comparison and
    // keeps the sum of products of 32-bit dimensions from overflowing.
    lastByteIndex = b.add(lastByteIndex, b.mul(last, b.umin(stride, limit)));
  }
  return b.and_(nonempty, b.icmp_uge(lastByteIndex, limit));
}

void initialize_tma(Location loc, ConversionPatternRewriter &rewriter,
                    const NVIDIA::TargetInfo &targetInfo, Value descPtr,
                    ArrayRef<uint32_t> words) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  constexpr int kWarpSize = 32;
  static_assert(std::tuple_size_v<TensormapWords> == kWarpSize);
  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));

  Value fillVal = b.i32_val(0);
  for (auto [i, word] : llvm::enumerate(words)) {
    if (word)
      fillVal =
          b.select(b.icmp_eq(threadId, b.i32_val(i)), b.i32_val(word), fillVal);
  }
  auto writeAddr =
      b.gep(descPtr.getType(), fillVal.getType(), descPtr, threadId);
  targetInfo.storeShared(rewriter, loc, writeAddr, fillVal, pred);
  LLVM::NVIDIA::createSyncWarp(loc, rewriter);
}

struct TensormapCreateOpConversion
    : public ConvertOpToLLVMPattern<ttng::TensormapCreateOp> {
  const NVIDIA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfo;

  TensormapCreateOpConversion(LLVMTypeConverter &converter,
                              const NVIDIA::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo),
        axisInfo(axisInfo) {}

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
    std::optional<TensormapTemplate> packed;
    if (capability == 90 || isBlackwell)
      packed = packTensormapTemplate(op, axisInfo, capability, ptxVersion);
    auto words =
        packed ? ArrayRef<uint32_t>(packed->words) : ArrayRef<uint32_t>{};
    initialize_tma(loc, rewriter, targetInfo, smemBase, words);
    Value isLeader = b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
    auto [previousBlock, updateBlock, continuationBlock] =
        createIfBlock(rewriter, loc, isLeader);
    (void)previousBlock;
    rewriter.setInsertionPointToStart(updateBlock);
    if (packed && packed->largeAlloc.value_or(false)) {
      // Preserve the prefilled class when patching the runtime address.
      PTXBuilder ptxBuilder;
      auto &replace =
          *ptxBuilder.create("tensormap.replace.tile.global_address.alloc_same."
                             "shared::cta.b1024.b64");
      replace(ptxBuilder.newAddrOperand(smemBase, "r"),
              ptxBuilder.newOperand(adaptor.getGlobalAddress(), "l"));
      ptxBuilder.launch(rewriter, loc, void_ty(ctx));
    } else {
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::GLOBAL_ADDRESS, smemBase,
                                adaptor.getGlobalAddress());
    }
    Value useLarge;
    if (isBlackwell && ptxVersion >= 95 && !(packed && packed->largeAlloc))
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
    if (!packed && op.getRank() != 1)
      tensormap_replace_generic(loc, ctx, rewriter, NVVM::TensormapField::RANK,
                                smemBase, b.i32_val(op.getRank() - 1));
    for (auto [i, dim] : llvm::enumerate(adaptor.getBoxDim())) {
      if (packed && packed->boxDims[i])
        continue;
      tensormap_replace_generic(
          loc, ctx, rewriter, NVVM::TensormapField::BOX_DIM, smemBase, dim, i);
    }
    for (auto [i, dim] : llvm::enumerate(adaptor.getGlobalDim())) {
      if (packed && packed->globalDims[i])
        continue;
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::GLOBAL_DIM, smemBase, dim,
                                i);
    }
    for (auto [i, strideVal] : llvm::enumerate(adaptor.getGlobalStride())) {
      if (packed && packed->globalStrides[i])
        continue;
      if (needsStrideWorkaround) {
        // Workaround for a ptxas bug
        strideVal = b.ashr(strideVal, b.i64_val(4));
      }
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::GLOBAL_STRIDE, smemBase,
                                strideVal, i);
    }
    for (auto [i, stride] : llvm::enumerate(adaptor.getElementStride())) {
      if (packed && packed->elementStrides[i])
        continue;
      tensormap_replace_generic(loc, ctx, rewriter,
                                NVVM::TensormapField::ELEMENT_STRIDE, smemBase,
                                stride, i);
    }
    if (!packed) {
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
              ctx,
              static_cast<NVVM::TensormapSwizzleMode>(op.getSwizzleMode())));
      tensormap_replace_attr(
          loc, ctx, rewriter, NVVM::TensormapField::FILL_MODE, smemBase,
          NVVM::TensormapFillModeAttr::get(
              ctx, static_cast<NVVM::TensormapFillMode>(op.getFillMode())));
    }
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
    ModuleAxisInfoAnalysis &axisInfo, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<TensormapCreateOpConversion>(typeConverter, targetInfo, axisInfo,
                                            benefit);
  patterns.add<TensormapFenceproxyAcquireOpConversion,
               ReinterpretTensorDescOpConversion>(typeConverter, benefit);
}
