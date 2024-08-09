#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_MATHTOLIBMVEC
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpT> struct VecOpToFp32 : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  VecOpToFp32(MLIRContext *context) : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isBF16() && !elemTy.isF16())
      return failure();

    Type fp32VecTy = vecTy.cloneWith(std::nullopt, rewriter.getF32Type());
    SmallVector<Value> fp32Ops;
    for (auto operand : op->getOperands())
      fp32Ops.push_back(
          rewriter.create<arith::ExtFOp>(loc, fp32VecTy, operand));
    auto newOp = rewriter.create<OpT>(loc, fp32VecTy, fp32Ops);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, vecTy, newOp);
    return success();
  }
};

// Decompose vector operation to singe-dimensional vector operations
// with a native AVX512 vector size.
template <typename OpT>
struct DecomposeToNativeVecs : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  DecomposeToNativeVecs(MLIRContext *context)
      : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    int64_t numElems = vecTy.getNumElements();
    if (numElems * elemTy.getIntOrFloatBitWidth() < 128)
      return failure();

    // Produce a new shape where trailing dimensions wouldn't exceed the native
    // vector size.
    auto shape = vecTy.getShape();
    SmallVector<int64_t> newShape(1, 1);
    int64_t elemsPerVec = 512 / elemTy.getIntOrFloatBitWidth();
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
      int64_t size = shape[i];
      if (newShape.size() > 1) {
        newShape.insert(newShape.begin(), size);
      } else {
        int64_t combined = newShape[0] * size;
        if (combined > elemsPerVec) {
          newShape[0] = elemsPerVec;
          newShape.insert(newShape.begin(), combined / elemsPerVec);
        } else {
          newShape[0] = combined;
        }
      }
    }
    if (newShape == shape)
      return failure();

    // Convert input operand to the new shape.
    SmallVector<Value> reshapedInputs;
    for (auto operand : op->getOperands()) {
      auto operandTy = cast<VectorType>(operand.getType());
      auto newOperandTy = VectorType::get(newShape, operandTy.getElementType());
      reshapedInputs.push_back(
          rewriter.create<vector::ShapeCastOp>(loc, newOperandTy, operand));
    }

    // Decompose the original operation to a set of operations on native
    // vectors.
    auto newOpTy = VectorType::get(newShape, elemTy);
    auto subResTy = VectorType::get(newShape.back(), elemTy);
    Value newRes = rewriter.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(newOpTy, rewriter.getFloatAttr(elemTy, 0)));
    auto strides = computeStrides(newShape);
    // Remove the last stride to produce sub-vector indices.
    strides.pop_back();
    for (int64_t idx = 0; idx < numElems; idx += newShape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(reshapedInputs.size());
      std::transform(reshapedInputs.begin(), reshapedInputs.end(),
                     subInputs.begin(), [&](auto val) {
                       return rewriter.create<vector::ExtractOp>(loc, val,
                                                                 indices);
                     });
      Value subRes = rewriter.create<OpT>(loc, subResTy, subInputs);
      newRes = rewriter.create<vector::InsertOp>(loc, subRes, newRes, indices);
    }

    // Reshape the result back to the original type.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecTy, newRes);
    return success();
  }
};

template <typename OpT>
struct VecOpToLibmvecCall : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  VecOpToLibmvecCall(MLIRContext *context, StringRef fp32FnBaseName,
                     StringRef fp64FnBaseName, bool use_sleef)
      : OpRewritePattern<OpT>(context) {
    this->fp32FnBaseName = fp32FnBaseName;
    this->fp64FnBaseName = fp64FnBaseName;
    this->use_sleef = use_sleef;
  }

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getRank() > 1)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    auto fnName = use_sleef
                      ? getSleefName(elemTy.isF32(), vecTy.getNumElements())
                      : getLibmvecName(elemTy.isF32(), vecTy.getNumElements(),
                                       op->getOperands());
    if (fnName.empty())
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, fnName));
    // Generate function declaration if it doesn't exists yet.
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      auto fnTy = FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc =
          rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), fnName, fnTy);
      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fnName, op.getType(),
                                              op->getOperands());
    return success();
  }

  std::string getLibmvecName(bool isFp32, int64_t numElems,
                             ValueRange ops) const {
    auto baseName = isFp32 ? fp32FnBaseName : fp64FnBaseName;
    int64_t vecSize = numElems * (isFp32 ? 32 : 64);
    std::string isaPrefix;
    if (vecSize == 128) {
      isaPrefix = "b";
    } else if (vecSize == 256) {
      isaPrefix = "d";
    } else if (vecSize == 512) {
      isaPrefix = "e";
    } else {
      return "";
    }
    std::string fnName = "_ZGV" + isaPrefix + "N" + std::to_string(numElems);
    for (auto operand : ops)
      fnName += "v";
    fnName += "_" + baseName;
    return fnName;
  }

  std::string getSleefName(bool isFp32, int64_t numElems) const {
    int64_t vecSize = numElems * (isFp32 ? 32 : 64);
    if (vecSize < 128)
      return "";
    auto baseName = isFp32 ? fp32FnBaseName : (fp64FnBaseName + "d");
    return "Sleef_" + baseName + std::to_string(numElems) + "_u10";
  }

private:
  std::string fp32FnBaseName;
  std::string fp64FnBaseName;
  bool use_sleef;
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns, StringRef fp32FnName,
                           StringRef fp64FnName, bool use_sleef) {
  patterns.add<VecOpToFp32<OpTy>>(patterns.getContext());
  patterns.add<DecomposeToNativeVecs<OpTy>>(patterns.getContext());
  patterns.add<VecOpToLibmvecCall<OpTy>>(patterns.getContext(), fp32FnName,
                                         fp64FnName, use_sleef);
}

struct MathToLibmvecPass
    : public mlir::triton::cpu::impl::MathToLibmvecBase<MathToLibmvecPass> {
  MathToLibmvecPass() = default;

  MathToLibmvecPass(bool use_sleef) { this->use_sleef = use_sleef; }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);
    populatePatternsForOp<math::AcosOp>(patterns, "acosf", "acos", use_sleef);
    populatePatternsForOp<math::AcoshOp>(patterns, "acoshf", "acosh",
                                         use_sleef);
    populatePatternsForOp<math::AsinOp>(patterns, "asinf", "asin", use_sleef);
    populatePatternsForOp<math::AsinhOp>(patterns, "asinhf", "asinh",
                                         use_sleef);
    populatePatternsForOp<math::AtanOp>(patterns, "atanf", "atan", use_sleef);
    populatePatternsForOp<math::AtanhOp>(patterns, "atanhf", "atanh",
                                         use_sleef);
    populatePatternsForOp<math::CbrtOp>(patterns, "cbrtf", "cbrt", use_sleef);
    populatePatternsForOp<math::CosOp>(patterns, "cosf", "cos", use_sleef);
    populatePatternsForOp<math::CoshOp>(patterns, "coshf", "cosh", use_sleef);
    populatePatternsForOp<math::ErfOp>(patterns, "erff", "erf", use_sleef);
    populatePatternsForOp<math::ExpOp>(patterns, "expf", "exp", use_sleef);
    populatePatternsForOp<math::Exp2Op>(patterns, "exp2f", "exp2", use_sleef);
    populatePatternsForOp<math::LogOp>(patterns, "logf", "log", use_sleef);
    populatePatternsForOp<math::Log2Op>(patterns, "log2f", "log2", use_sleef);
    populatePatternsForOp<math::Log10Op>(patterns, "log10f", "log10",
                                         use_sleef);
    populatePatternsForOp<math::Log1pOp>(patterns, "log1pf", "log1p",
                                         use_sleef);
    populatePatternsForOp<math::SinOp>(patterns, "sinf", "sin", use_sleef);
    populatePatternsForOp<math::SinhOp>(patterns, "sinhf", "sinh", use_sleef);
    populatePatternsForOp<math::TanOp>(patterns, "tanf", "tan", use_sleef);
    populatePatternsForOp<math::TanhOp>(patterns, "tanhf", "tanh", use_sleef);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMathToLibmvecPass() {
  return std::make_unique<MathToLibmvecPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createMathToLibmvecPass(bool use_sleef) {
  return std::make_unique<MathToLibmvecPass>(use_sleef);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
