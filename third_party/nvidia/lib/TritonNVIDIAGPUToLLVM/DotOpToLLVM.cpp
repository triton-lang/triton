#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = ::mlir::triton::gpu;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring,
                         bool isHopperF64);

LogicalResult convertMMADotScaled(triton::DotScaledOp op,
                                  triton::DotScaledOp::Adaptor adaptor,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter);

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);

namespace {
struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern<triton::DotScaledOp>::ConvertOpToLLVMPattern;

  ScaledDotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotScaledOp>(converter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, triton::DotScaledOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return convertMMADotScaled(op, adaptor, getTypeConverter(), rewriter);
  }

private:
  int computeCapability;
};

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    NvidiaMmaEncodingAttr mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (mmaLayout) {
      if (mmaLayout.getVersionMajor() == 2) {
        bool isHopperF64 =
            computeCapability == 90 &&
            cast<RankedTensorType>(A.getType()).getElementType().isF64();
        return convertMMA(op, adaptor, getTypeConverter(), rewriter,
                          mmaLayout.isTuring(), isHopperF64);
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

private:
  int computeCapability;
};

struct WarpGroupDotOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    return convertWGMMA(op, adaptor, getTypeConverter(), rewriter,
                        getThreadId(rewriter, loc));
  }
};

struct WarpGroupDotWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotWaitOp>::ConvertOpToLLVMPattern;

  static FailureOr<Value> packInputs(Location loc, RewriterBase &rewriter,
                                     ValueRange inputs) {
    if (inputs.size() == 1) {
      return inputs[0];
    }
    SmallVector<Type> types;
    // Pack the inputs into a single struct.
    for (Type type : inputs.getTypes()) {
      auto structType = dyn_cast<LLVM::LLVMStructType>(type);
      if (!structType)
        return failure();
      llvm::append_range(types, structType.getBody());
    }
    auto packedType =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    Value packed = LLVM::UndefOp::create(rewriter, loc, packedType);
    unsigned outputStructIndex = 0;
    for (Value input : inputs) {
      for (auto [i, type] : llvm::enumerate(
               cast<LLVM::LLVMStructType>(input.getType()).getBody())) {
        Value value = LLVM::ExtractValueOp::create(rewriter, loc, input, i);
        packed = LLVM::InsertValueOp::create(rewriter, loc, packedType, packed,
                                             value, outputStructIndex++);
      }
    }
    return packed;
  }

  static SmallVector<Value> unpackOutput(Location loc, RewriterBase &rewriter,
                                         Value packedOutput, TypeRange types) {
    SmallVector<Value> outputs;
    if (types.size() == 1) {
      outputs.push_back(packedOutput);
      return outputs;
    }
    // Unpack the output into the original struct types.
    unsigned outputStructIndex = 0;
    for (Type type : types) {
      auto structType = cast<LLVM::LLVMStructType>(type);
      Value unpacked = LLVM::UndefOp::create(rewriter, loc, structType);
      for (auto [i, type] : llvm::enumerate(structType.getBody())) {
        Value value = LLVM::ExtractValueOp::create(rewriter, loc, packedOutput,
                                                   outputStructIndex++);
        unpacked = LLVM::InsertValueOp::create(rewriter, loc, structType,
                                               unpacked, value, i);
      }
      outputs.push_back(unpacked);
    }
    return outputs;
  }

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    Location loc = op.getLoc();
    ValueRange inputs = adaptor.getInputs();
    auto packed = packInputs(loc, rewriter, inputs);
    if (failed(packed))
      return failure();
    auto waitOp = triton::nvgpu::WGMMAWaitGroupOp::create(rewriter, loc,
                                                          *packed, pendings);
    auto outputs =
        unpackOutput(loc, rewriter, waitOp.getResult(), inputs.getTypes());
    rewriter.replaceOp(op, outputs);

    // When there are multiple warp groups, we need a barrier to ensure all warp
    // groups have finished.
    if (!op.getWarpGroupLocal() && ttg::lookupNumWarps(op) > 4) {
      triton::gpu::BarrierOp::create(rewriter, loc,
                                     triton::gpu::AddrSpace::Local);
    }
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, computeCapability, benefit);
  patterns.add<WarpGroupDotOpConversion>(typeConverter, benefit);
  patterns.add<WarpGroupDotWaitOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, computeCapability,
                                      benefit);
}
