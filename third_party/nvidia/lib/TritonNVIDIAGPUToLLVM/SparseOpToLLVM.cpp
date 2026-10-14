#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::NVIDIA;

namespace {

static constexpr std::array<int, 7> spDecompressNumChoices = {64, 32, 16, 8,
                                                              4,  2,  1};

SmallVector<Value> createSpDecompress(Location loc, ArrayRef<Value> mdataRegs,
                                      ArrayRef<Value> cdataRegs, int num,
                                      int elemBitWidth,
                                      ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  assert(num == 1 || num == 2 || num == 4 || num == 8 || num == 16 ||
         num == 32 || num == 64);
  assert((elemBitWidth == 8 || elemBitWidth == 16) &&
         "unsupported spdecompress element size");
  int cdataRegsExpected = (2 * elemBitWidth * num + 31) / 32;
  int numMDataRegs = (2 * 2 * num + 31) / 32;
  assert(cdataRegs.size() == static_cast<size_t>(cdataRegsExpected));
  assert(mdataRegs.size() == static_cast<size_t>(numMDataRegs));

  int numDataRegs = (4 * elemBitWidth * num + 31) / 32;
  PTXBuilder ptxBuilder;
  std::string opcode = "spdecompress.b" + std::to_string(elemBitWidth) +
                       ".b2.sp::2:4.x" + std::to_string(num) + " {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numDataRegs; ++i) {
    if (i > 0)
      opcode += ", ";
    opcode += "$" + std::to_string(operands.size());
    operands.push_back(ptxBuilder.newOperand("=r"));
  }
  opcode += "}, {";
  for (size_t i = 0; i < mdataRegs.size(); ++i) {
    if (i > 0)
      opcode += ", ";
    opcode += "$" + std::to_string(operands.size());
    operands.push_back(ptxBuilder.newOperand(mdataRegs[i], "r"));
  }
  opcode += "}, {";
  for (size_t i = 0; i < cdataRegs.size(); ++i) {
    if (i > 0)
      opcode += ", ";
    opcode += "$" + std::to_string(operands.size());
    operands.push_back(ptxBuilder.newOperand(cdataRegs[i], "r"));
  }
  opcode += "};";

  auto &spdecompress = *ptxBuilder.create(opcode);
  spdecompress(operands, /*onlyAttachMLIRArgs=*/true);

  Type retTy = i32_ty;
  if (numDataRegs > 1)
    retTy = struct_ty(SmallVector<Type>(numDataRegs, i32_ty));
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> dataRegs;
  dataRegs.reserve(numDataRegs);
  if (numDataRegs == 1) {
    dataRegs.push_back(ret);
    return dataRegs;
  }
  for (int i = 0; i < numDataRegs; ++i)
    dataRegs.push_back(b.extract_val(i32_ty, ret, i));
  return dataRegs;
}

struct SpDecompressOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::SpDecompressOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::SpDecompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<Value> cdataElems =
        unpackUniqueTensorElements(loc, adaptor.getCdata(), rewriter);
    SmallVector<Value> mdataRegs =
        unpackUniqueTensorElements(loc, adaptor.getMdata(), rewriter);
    if (cdataElems.empty())
      return rewriter.notifyMatchFailure(op, "empty cdata vector");
    if (mdataRegs.empty())
      return rewriter.notifyMatchFailure(op, "empty mdata vector");

    int elemBitWidth = resultTy.getElementType().getIntOrFloatBitWidth();
    if (elemBitWidth != 8 && elemBitWidth != 16) {
      return rewriter.notifyMatchFailure(op, "unsupported spdecompress type");
    }
    int elemsPerReg = 32 / elemBitWidth;
    if (cdataElems.size() % elemsPerReg != 0) {
      return rewriter.notifyMatchFailure(
          op, "cdata elements per thread must be divisible by elements per "
              "32-bit register");
    }

    SmallVector<Value> cdataRegs =
        LLVM::NVIDIA::packValues(cdataElems, i32_ty, loc, rewriter);
    if (cdataRegs.empty())
      return rewriter.notifyMatchFailure(op, "failed to pack cdata");

    auto chooseNum = [](int64_t remaining) -> int {
      for (int candidate : spDecompressNumChoices) {
        if (remaining >= candidate)
          return candidate;
      }
      return 0;
    };

    size_t cdataRegOffset = 0;
    size_t mdataRegOffset = 0;
    int repeatsPerReg = 16 / elemBitWidth;
    SmallVector<Value> denseRegs;
    denseRegs.reserve(cdataRegs.size() * 2);
    while (cdataRegOffset < cdataRegs.size()) {
      int64_t remainingRegs = cdataRegs.size() - cdataRegOffset;
      int64_t remainingRepeats = remainingRegs * repeatsPerReg;
      int num = chooseNum(remainingRepeats);
      if (num == 0) {
        return rewriter.notifyMatchFailure(
            op, "unable to choose a valid spdecompress .num");
      }

      size_t numCdataRegs = (2 * elemBitWidth * num + 31) / 32;
      if (cdataRegOffset + numCdataRegs > cdataRegs.size()) {
        return rewriter.notifyMatchFailure(op,
                                           "insufficient cdata registers for "
                                           "spdecompress chunk");
      }
      size_t numMdataRegs = (2 * 2 * num + 31) / 32;
      if (mdataRegOffset + numMdataRegs > mdataRegs.size()) {
        return rewriter.notifyMatchFailure(op,
                                           "insufficient mdata registers for "
                                           "spdecompress chunk");
      }

      ArrayRef<Value> cdataChunk(cdataRegs.data() + cdataRegOffset,
                                 numCdataRegs);
      ArrayRef<Value> mdataChunk(mdataRegs.data() + mdataRegOffset,
                                 numMdataRegs);
      SmallVector<Value> denseChunk = createSpDecompress(
          loc, mdataChunk, cdataChunk, num, elemBitWidth, rewriter);
      denseRegs.append(denseChunk.begin(), denseChunk.end());
      cdataRegOffset += numCdataRegs;
      mdataRegOffset += numMdataRegs;
    }

    if (mdataRegOffset != mdataRegs.size()) {
      return rewriter.notifyMatchFailure(
          op, "unused mdata registers remain after chunked spdecompress");
    }

    Type llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Value> denseElems =
        LLVM::NVIDIA::unpackValues(denseRegs, llvmElemTy, loc, rewriter);
    Value resultStruct =
        packUniqueTensorElements(loc, getTypeConverter(), denseElems, rewriter,
                                 op.getResult().getType());
    rewriter.replaceOp(op, resultStruct);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateSparseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<SpDecompressOpConversion>(typeConverter, benefit);
}
