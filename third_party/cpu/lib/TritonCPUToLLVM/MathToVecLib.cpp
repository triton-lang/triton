#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_MATHTOVECLIB
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
          arith::ExtFOp::create(rewriter, loc, fp32VecTy, operand));
    auto newOp = OpT::create(rewriter, loc, fp32VecTy, fp32Ops);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, vecTy, newOp);
    return success();
  }
};

// Decompose vector operation to single-dimensional vector operations
// with a AVX512 for x86 or NEON for ARM.
template <typename OpT>
struct DecomposeToNativeVecs : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;
  // CPU SIMD vector size in bits
  size_t vec_bits;

  DecomposeToNativeVecs(MLIRContext *context,
                        size_t native_vec_size_in_bits = 512)
      : OpRewritePattern<OpT>(context), vec_bits(native_vec_size_in_bits) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.isScalable())
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
    int64_t elemsPerVec = vec_bits / elemTy.getIntOrFloatBitWidth();
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
          vector::ShapeCastOp::create(rewriter, loc, newOperandTy, operand));
    }

    // Decompose the original operation to a set of operations on native
    // vectors.
    auto newOpTy = VectorType::get(newShape, elemTy);
    auto subResTy = VectorType::get(newShape.back(), elemTy);
    Value newRes = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(newOpTy, rewriter.getFloatAttr(elemTy, 0)));
    auto strides = computeStrides(newShape);
    // Remove the last stride to produce sub-vector indices.
    strides.pop_back();
    for (int64_t idx = 0; idx < numElems; idx += newShape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(reshapedInputs.size());
      std::transform(reshapedInputs.begin(), reshapedInputs.end(),
                     subInputs.begin(), [&](auto val) {
                       return vector::ExtractOp::create(rewriter, loc, val,
                                                        indices);
                     });
      Value subRes =
          OpT::create(rewriter, loc, subResTy, subInputs, op->getAttrs());
      newRes = vector::InsertOp::create(rewriter, loc, subRes, newRes, indices);
    }

    // Reshape the result back to the original type.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecTy, newRes);
    return success();
  }
};

using ExternElementwiseOp = triton::cpu::ExternElementwiseOp;

/*
 * libsleef does not contain implementations for 2-element f32 vectors, so we
 * pad any such vectors to size 4 instead. For f64 the minimum vector size is 2.
 */
struct PadSmallVecsForSleef : public OpRewritePattern<ExternElementwiseOp> {
public:
  using OpRewritePattern<ExternElementwiseOp>::OpRewritePattern;

  PadSmallVecsForSleef(MLIRContext *context)
      : OpRewritePattern<ExternElementwiseOp>(context) {}

  LogicalResult matchAndRewrite(ExternElementwiseOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.isScalable())
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    int64_t minVecSize = elemTy.isF32() ? 4 : 2;
    int64_t numElems = vecTy.getNumElements();
    if (numElems >= minVecSize)
      return failure();

    // Create a single-element vector for shuffle to use
    auto paddingVec = vector::BroadcastOp::create(
        rewriter, loc, VectorType::get({1}, elemTy), b.undef(elemTy));
    // Assign indices such that shuffle will pad the original vector with
    // elements from the paddingVec
    SmallVector<int64_t> indices(minVecSize);
    for (int i = 0; i < minVecSize; ++i) {
      if (i < numElems)
        indices[i] = i;
      else
        indices[i] = numElems;
    }
    SmallVector<Value> newOperands;
    for (auto argVal : op.getOperands()) {
      auto shuf =
          vector::ShuffleOp::create(rewriter, loc, argVal, paddingVec, indices);
      newOperands.push_back(shuf.getResult());
    }
    // Update return type of extern call
    auto newVecTy = VectorType::get({minVecSize}, elemTy);
    auto extern_elem = ExternElementwiseOp::create(
        rewriter, loc, newVecTy, newOperands, op.getSymbol(), op.getPure());
    indices.resize(numElems);
    // Truncate result to original size
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, extern_elem.getResult(),
                                                   paddingVec, indices);
    return success();
  }
};

using GetVecFnNameFn = std::function<std::string(
    unsigned /*bitwidth*/, unsigned /*numel*/, ValueRange /*operands*/)>;

class MvecNameGenerator {
public:
  // The ulp and useSve parameters are accepted for API compatibility with
  // SleefNameGenerator but are not used by Mvec.
  explicit MvecNameGenerator(StringRef baseName, unsigned ulp, bool useSve)
      : baseName(baseName) {}

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange operands) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
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
    std::string fnName = "_ZGV" + isaPrefix + "N" + std::to_string(numel);
    for (auto operand : operands)
      fnName += "v";
    return fnName + "_" + baseName + (bitwidth == 32 ? "f" : "");
  }

private:
  std::string baseName;
};

class SleefNameGenerator {
public:
  SleefNameGenerator(StringRef baseName, unsigned ulp, bool useSve)
      : baseName(baseName), ulpSuffix(4, '\0'), useSve(useSve) {
    if (ulp == 0) {
      ulpSuffix = "";
    } else {
      char buf[13]; // "_u" + 10 unsigned digits + '\0'
      snprintf(buf, sizeof(buf), "_u%02u", ulp);
      ulpSuffix = buf;
    }
  }

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange /*operands*/) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    if (vecSize < 128)
      return "";
    if (useSve) {
      // SLEEF SVE functions use a different naming convention:
      //   Sleef_<baseName>dx_u<ulp>sve  (double, bitwidth 64, ulp > 0)
      //   Sleef_<baseName>fx_u<ulp>sve  (float, bitwidth 32, ulp > 0)
      //   Sleef_<baseName>dx_sve        (double, bitwidth 64, ulp == 0)
      //   Sleef_<baseName>fx_sve        (float, bitwidth 32, ulp == 0)
      // No element count is included because SVE is a VLA ISA.
      std::string sveSuffix = ulpSuffix.empty() ? "_sve" : ulpSuffix + "sve";
      return "Sleef_" + baseName + (bitwidth == 32 ? "fx" : "dx") + sveSuffix;
    }
    return "Sleef_" + baseName + (bitwidth == 32 ? "f" : "d") +
           std::to_string(numel) + ulpSuffix;
  }

private:
  std::string baseName;
  std::string ulpSuffix;
  bool useSve;
};

template <typename OpT>
struct OpToVecLibConversion : public OpRewritePattern<OpT> {
public:
  OpToVecLibConversion(MLIRContext *context, bool useSve = false)
      : OpRewritePattern<OpT>(context), useSve(useSve) {}

  virtual std::string getVecFnName(OpT op, unsigned bitwidth,
                                   unsigned numel) const = 0;
  virtual std::string getScalarFnName(OpT op, unsigned bitwidth) const {
    return {};
  };

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Type ty = op.getType();
    std::string fnName;

    bool lowerToSve;
    VectorType vecTy = dyn_cast<VectorType>(ty);
    if (vecTy && !vecTy.isScalable() && (useSve || vecTy.getRank() == 1)) {
      fnName = getVecFnName(op, vecTy.getElementTypeBitWidth(),
                            vecTy.getNumElements());
      lowerToSve = useSve;
    } else if (ty.isF32() || ty.isF64()) {
      fnName = getScalarFnName(op, ty.getIntOrFloatBitWidth());
      lowerToSve = false;
    }

    if (fnName.empty())
      return failure();

    auto func = op->template getParentOfType<FunctionOpInterface>();
    if (lowerToSve && !func)
      return failure();

    // A SLEEF SVE function takes one full scalable register, regardless of the
    // fixed-width shape of the original Triton block.
    Type callTy = ty;
    if (lowerToSve)
      callTy = VectorType::get({128 / vecTy.getElementTypeBitWidth()},
                               vecTy.getElementType(), {true});

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, fnName));
    // Generate function declaration if it doesn't exists yet.
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      FunctionType fnTy;
      if (lowerToSve) {
        SmallVector<Type> inputTypes(op->getNumOperands(), callTy);
        fnTy = FunctionType::get(rewriter.getContext(), inputTypes, {callTy});
      } else {
        fnTy = FunctionType::get(rewriter.getContext(), op->getOperandTypes(),
                                 {callTy});
      }
      opFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(), fnName,
                                    fnTy);
      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }

    if (lowerToSve) {
      Location loc = op.getLoc();
      int64_t numElems = vecTy.getNumElements();
      auto flatTy = VectorType::get({numElems}, vecTy.getElementType());
      auto bufferTy = MemRefType::get({numElems}, vecTy.getElementType());
      SmallVector<Value> inputBuffers;
      auto pointerTy = LLVM::LLVMPointerType::get(rewriter.getContext());
      // Restore the stack after each evaluation, including inside loops, so
      // scratch storage cannot accumulate across operations or iterations.
      Value stack = LLVM::StackSaveOp::create(rewriter, loc, pointerTy);
      for (unsigned i = 0; i < op->getNumOperands(); ++i)
        inputBuffers.push_back(
            memref::AllocaOp::create(rewriter, loc, bufferTy));
      Value resultBuffer = memref::AllocaOp::create(rewriter, loc, bufferTy);

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upper = arith::ConstantIndexOp::create(rewriter, loc, numElems);
      Value lanes = arith::ConstantIndexOp::create(
          rewriter, loc, 128 / vecTy.getElementTypeBitWidth());
      Value vscale = vector::VectorScaleOp::create(rewriter, loc);
      Value step = arith::MulIOp::create(rewriter, loc, lanes, vscale);
      for (auto [input, buffer] : llvm::zip(op->getOperands(), inputBuffers)) {
        Value flat = vector::ShapeCastOp::create(rewriter, loc, flatTy, input);
        vector::StoreOp::create(rewriter, loc, flat, buffer, ValueRange{zero});
      }

      auto scalableTy = cast<VectorType>(callTy);
      auto maskTy = scalableTy.cloneWith(std::nullopt, rewriter.getI1Type());
      Value passthru = arith::ConstantOp::create(
          rewriter, loc,
          SplatElementsAttr::get(scalableTy,
                                 rewriter.getZeroAttr(vecTy.getElementType())));
      auto loop = scf::ForOp::create(rewriter, loc, zero, upper, step);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());
        Value index = loop.getInductionVar();
        Value remaining = arith::SubIOp::create(rewriter, loc, upper, index);
        Value mask = vector::CreateMaskOp::create(rewriter, loc, maskTy,
                                                  ValueRange{remaining});
        SmallVector<Value> inputs;
        for (Value buffer : inputBuffers)
          inputs.push_back(
              vector::MaskedLoadOp::create(rewriter, loc, scalableTy, buffer,
                                           ValueRange{index}, mask, passthru));
        Value result = func::CallOp::create(rewriter, loc, fnName,
                                            TypeRange{callTy}, inputs)
                           .getResult(0);
        // SLEEF evaluates every lane. Only the active results are written back.
        vector::MaskedStoreOp::create(rewriter, loc, resultBuffer,
                                      ValueRange{index}, mask, result);
      }
      Value result = vector::LoadOp::create(rewriter, loc, flatTy, resultBuffer,
                                            ValueRange{zero});
      // The loaded SSA value remains valid after its scratch storage is freed.
      LLVM::StackRestoreOp::create(rewriter, loc, stack);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecTy, result);
      return success();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fnName, ty,
                                              op->getOperands());
    return success();
  }

private:
  bool useSve;
};

template <typename OpT>
struct VecOpToVecLibConversion : public OpToVecLibConversion<OpT> {
public:
  VecOpToVecLibConversion(MLIRContext *context, GetVecFnNameFn getVecFnName,
                          bool useSve)
      : OpToVecLibConversion<OpT>(context, useSve),
        getVecFnNameImpl(getVecFnName) {}

  std::string getVecFnName(OpT op, unsigned bitwidth,
                           unsigned numel) const override {
    return getVecFnNameImpl(bitwidth, numel, op->getOperands());
  }

private:
  GetVecFnNameFn getVecFnNameImpl;
};

struct ExternElementwiseOpConversion
    : public OpToVecLibConversion<triton::cpu::ExternElementwiseOp> {
  using OpToVecLibConversion::OpToVecLibConversion;

  std::string getVecFnName(triton::cpu::ExternElementwiseOp op,
                           unsigned bitwidth, unsigned numel) const override {
    auto fnName = op.getSymbol();
    auto numelIdx = fnName.find("%(numel)");
    if (numelIdx == StringRef::npos)
      return fnName.str();
    return (fnName.take_front(numelIdx) + Twine(numel) +
            fnName.drop_front(numelIdx + 8))
        .str();
  }

  std::string getScalarFnName(triton::cpu::ExternElementwiseOp op,
                              unsigned bitwidth) const override {
    auto fnName = op.getSymbol();
    auto numelIdx =
        bitwidth == 64 ? fnName.find("d%(numel)") : fnName.find("f%(numel)");
    if (numelIdx == StringRef::npos)
      return fnName.str();
    // Scalar 64-bit functions don't have a 'd' infix in their name.
    return (fnName.take_front(numelIdx) + Twine(bitwidth == 64 ? "" : "f") +
            fnName.drop_front(numelIdx + 9))
        .str();
  }
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns,
                           GetVecFnNameFn getVecFnName,
                           size_t vec_size_in_bits = 512, bool useSve = false) {
  patterns.add<VecOpToFp32<OpTy>>(patterns.getContext());
  if (!useSve)
    patterns.add<DecomposeToNativeVecs<OpTy>>(patterns.getContext(),
                                              vec_size_in_bits);
  patterns.add<VecOpToVecLibConversion<OpTy>>(patterns.getContext(),
                                              getVecFnName, useSve);
}

struct MathToVecLibPass
    : public mlir::triton::cpu::impl::MathToVecLibBase<MathToVecLibPass> {
  MathToVecLibPass() = default;
  // Default to 128-bit if no features are specified.
  size_t vec_size_in_bits = 128;

  explicit MathToVecLibPass(VecLib lib, std::set<std::string> cpu_features) {
    this->lib = lib;
    // Store cpu_features in the base class member so that runOnOperation
    // can access them for SVE detection.
    this->cpu_features.clear();
    for (const auto &f : cpu_features)
      this->cpu_features.push_back(f);
    update_vec_size(cpu_features);
  }

  void update_vec_size(std::set<std::string> &cpu_features) {
    // Fixed-width calls (including explicit extern calls) use native SIMD
    // widths. SLEEF SVE math bypasses this decomposition and uses
    // vector.vscale.
    for (auto feature : cpu_features) {
      if (feature == "avx512f") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 512);
      } else if (feature == "avx") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 256);
      } else if (feature == "sse") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 128);
      } else if (feature == "sve" || feature == "sve2") {
        // Explicit fixed-width extern calls retain the 128-bit fallback.
        // SLEEF SVE math instead uses scalable vectors and a runtime-sized
        // loop.
        vec_size_in_bits = 128;
        break;
      } else if (feature == "neon") {
        // Arm NEON is fixed 128-bit SIMD ISA.
        vec_size_in_bits = 128;
        break;
      }
    }
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);

    if (!cpu_features.empty()) {
      std::set<std::string> cpu_features_set{cpu_features.begin(),
                                             cpu_features.end()};
      update_vec_size(cpu_features_set);
    }

    // Check whether SVE is available to select the appropriate SLEEF naming
    // convention.  SLEEF provides SVE-specific functions with the "sve"
    // suffix that use scalable vector types (VLA).
    bool useSve = false;
    if (!cpu_features.empty()) {
      std::set<std::string> cpu_features_set{cpu_features.begin(),
                                             cpu_features.end()};
      useSve = cpu_features_set.count("sve") > 0 ||
               cpu_features_set.count("sve2") > 0;
    }

    switch (lib) {
    case VecLib::Mvec: {
      populateCommonPatterns<MvecNameGenerator>(patterns);
      break;
    }
    case VecLib::Sleef: {
      populateCommonPatterns<SleefNameGenerator>(patterns, /*ulp=*/10, useSve);
      populatePatternsForOp<math::ExpM1Op>(
          patterns, SleefNameGenerator("expm1", /*ulp=*/10, useSve),
          vec_size_in_bits, useSve);
      populatePatternsForOp<math::FloorOp>(
          patterns, SleefNameGenerator("floor", /*ulp=*/0, useSve),
          vec_size_in_bits, useSve);
      populatePatternsForOp<math::SqrtOp>(
          patterns, SleefNameGenerator("sqrt", /*ulp=*/5, useSve),
          vec_size_in_bits, useSve);
      populatePatternsForOp<math::TruncOp>(
          patterns, SleefNameGenerator("trunc", /*ulp=*/0, useSve),
          vec_size_in_bits, useSve);
      break;
    }
    }

    patterns.add<DecomposeToNativeVecs<ExternElementwiseOp>>(
        patterns.getContext(), vec_size_in_bits);
    patterns.add<PadSmallVecsForSleef>(patterns.getContext());
    patterns.add<ExternElementwiseOpConversion>(patterns.getContext());

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }

  template <typename VecFnNameGenerator>
  void populateCommonPatterns(RewritePatternSet &patterns, unsigned ulp = 10,
                              bool useSve = false) const {
    populatePatternsForOp<math::AcosOp>(patterns,
                                        VecFnNameGenerator("acos", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::AcoshOp>(
        patterns, VecFnNameGenerator("acosh", ulp, useSve), vec_size_in_bits,
        useSve);
    populatePatternsForOp<math::AsinOp>(patterns,
                                        VecFnNameGenerator("asin", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::AsinhOp>(
        patterns, VecFnNameGenerator("asinh", ulp, useSve), vec_size_in_bits,
        useSve);
    populatePatternsForOp<math::AtanOp>(patterns,
                                        VecFnNameGenerator("atan", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::AtanhOp>(
        patterns, VecFnNameGenerator("atanh", ulp, useSve), vec_size_in_bits,
        useSve);
    populatePatternsForOp<math::CbrtOp>(patterns,
                                        VecFnNameGenerator("cbrt", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::CosOp>(patterns,
                                       VecFnNameGenerator("cos", ulp, useSve),
                                       vec_size_in_bits, useSve);
    populatePatternsForOp<math::CoshOp>(patterns,
                                        VecFnNameGenerator("cosh", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::ErfOp>(patterns,
                                       VecFnNameGenerator("erf", ulp, useSve),
                                       vec_size_in_bits, useSve);
    populatePatternsForOp<math::ExpOp>(patterns,
                                       VecFnNameGenerator("exp", ulp, useSve),
                                       vec_size_in_bits, useSve);
    populatePatternsForOp<math::Exp2Op>(patterns,
                                        VecFnNameGenerator("exp2", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::LogOp>(patterns,
                                       VecFnNameGenerator("log", ulp, useSve),
                                       vec_size_in_bits, useSve);
    populatePatternsForOp<math::Log2Op>(patterns,
                                        VecFnNameGenerator("log2", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::Log10Op>(
        patterns, VecFnNameGenerator("log10", ulp, useSve), vec_size_in_bits,
        useSve);
    populatePatternsForOp<math::Log1pOp>(
        patterns, VecFnNameGenerator("log1p", ulp, useSve), vec_size_in_bits,
        useSve);
    populatePatternsForOp<math::SinOp>(patterns,
                                       VecFnNameGenerator("sin", ulp, useSve),
                                       vec_size_in_bits, useSve);
    populatePatternsForOp<math::SinhOp>(patterns,
                                        VecFnNameGenerator("sinh", ulp, useSve),
                                        vec_size_in_bits, useSve);
    populatePatternsForOp<math::TanOp>(patterns,
                                       VecFnNameGenerator("tan", ulp, useSve),
                                       vec_size_in_bits, useSve);
    populatePatternsForOp<math::TanhOp>(patterns,
                                        VecFnNameGenerator("tanh", ulp, useSve),
                                        vec_size_in_bits, useSve);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createMathToVecLibPass(VecLib lib, std::set<std::string> cpu_features) {
  return std::make_unique<MathToVecLibPass>(lib, cpu_features);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
