#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;

namespace {

using ValueTableFMA = std::map<std::tuple<int, int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, int batch, int nonK, int K,
                           ConversionPatternRewriter &rewriter, Location loc) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(elems.size() == K * nonK * batch);
  int index = 0;
  for (unsigned b = 0; b < batch; ++b)
    for (unsigned k = 0; k < K; ++k)
      for (unsigned i = 0; i < nonK; ++i)
        res[{b, i, k}] = elems[index++];
  return res;
}

struct DotIntrinsic {
  int vectorSize;
  Type outElemTy;
  StringRef intrinsicName;
  SmallVector<Value> additionalArgs;
};

DotIntrinsic chooseIntrinsic(ConversionPatternRewriter &rewriter, Location loc,
                             triton::DotOp op) {
  auto aOpTy = cast<RankedTensorType>(op.getA().getType());
  auto aElemTy = aOpTy.getElementType();
  auto dOpTy = cast<RankedTensorType>(op.getD().getType());
  auto dElemTy = dOpTy.getElementType();
  auto mod = op->getParentOfType<ModuleOp>();
  auto arch = getAMDArch(mod);
  DotIntrinsic chosenOp;
  bool dotAvailable = arch == "gfx908" || arch == "gfx90a" ||
                      arch.starts_with("gfx94") || arch.starts_with("gfx11") ||
                      arch.starts_with("gfx103");
  if (dotAvailable) {
    if (aElemTy.isF16() && dElemTy.isF32()) {
      chosenOp.vectorSize = 2;
      chosenOp.outElemTy = f32_ty;
      chosenOp.intrinsicName = "llvm.amdgcn.fdot2";
      chosenOp.additionalArgs = {false_val()};
      return chosenOp;
    }
    if (aElemTy.isInteger(8) && dElemTy.isInteger(32)) {
      chosenOp.vectorSize = 4;
      chosenOp.outElemTy = i32_ty;
      chosenOp.intrinsicName = "llvm.amdgcn.sdot4";
      chosenOp.additionalArgs = {false_val()};
      return chosenOp;
    }
  }
  // choose one of FMA intrinsics
  assert(aElemTy.isIntOrFloat() && !aElemTy.isIntOrIndex());
  assert(aElemTy == dElemTy);
  assert(cast<RankedTensorType>(op.getA().getType()).getElementType() ==
         dElemTy);
  chosenOp.vectorSize = 1;
  chosenOp.outElemTy = aElemTy;
  if (aElemTy.isF32())
    chosenOp.intrinsicName = "llvm.fmuladd.f32";
  if (aElemTy.isF16())
    chosenOp.intrinsicName = "llvm.fmuladd.f16";
  chosenOp.additionalArgs = {};
  return chosenOp;
}

Value packOperand(ConversionPatternRewriter &rewriter, Location loc,
                  ValueTableFMA scalarValues, unsigned b, unsigned nonK,
                  unsigned k, unsigned vectorSize) {
  if (vectorSize == 1)
    return scalarValues[{b, nonK, k}];
  auto elemTy = scalarValues[{b, nonK, k}].getType();
  auto vecTy = vec_ty(elemTy, vectorSize);
  Value vec = undef(vecTy);
  for (int elem = 0; elem < vectorSize; ++elem) {
    vec = insert_element(vecTy, vec, scalarValues[{b, nonK, k + elem}],
                         i32_val(elem));
  }
  if (elemTy.isInteger(8)) {
    assert(vectorSize == 4);
    vec = bitcast(vec, i32_ty);
  }
  return vec;
}

Value generateDotOp(ConversionPatternRewriter &rewriter, Location loc,
                    DotIntrinsic op, Value a, Value b, Value c) {
  SmallVector<Value> args{a, b, c};
  args.append(op.additionalArgs.begin(), op.additionalArgs.end());
  SmallVector<Type> argTypes;
  for (auto arg : args)
    argTypes.push_back(arg.getType());
  auto funcType = LLVM::LLVMFunctionType::get(op.outElemTy, argTypes);
  auto d = call_intrinsic(op.outElemTy, op.intrinsicName, args);
  return d.getResult(0);
}

} // namespace

namespace mlir::triton::AMD {

LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());
  auto dElemTy = dTensorTy.getElementType();

  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));

  int K = aShapePerCTA[2];

  unsigned retSize[3];
  for (int i = 0; i < 3; ++i) {
    unsigned numRep = dShapePerCTA[i] / shapePerCTATile[i];
    numRep = std::max(static_cast<unsigned>(1), numRep);
    retSize[i] = numRep * sizePerThread[i];
  }

  auto has =
      getValueTableFromStructFMA(llA, retSize[0], retSize[1], K, rewriter, loc);
  auto hbs =
      getValueTableFromStructFMA(llB, retSize[0], retSize[2], K, rewriter, loc);

  SmallVector<Value> ret = cc;
  auto selectedOp = chooseIntrinsic(rewriter, loc, op);

  for (unsigned b = 0; b < retSize[0]; ++b)
    for (unsigned m = 0; m < retSize[1]; ++m)
      for (unsigned n = 0; n < retSize[2]; ++n) {
        unsigned idx[] = {b, m, n};
        unsigned linearIdx = 0;
        for (auto dim : llvm::reverse(order)) {
          linearIdx = linearIdx * retSize[dim] + idx[dim];
        }
        for (unsigned k = 0; k < K; k += selectedOp.vectorSize) {
          auto aOp =
              packOperand(rewriter, loc, has, b, m, k, selectedOp.vectorSize);
          auto bOp =
              packOperand(rewriter, loc, hbs, b, n, k, selectedOp.vectorSize);
          ret[linearIdx] = generateDotOp(rewriter, loc, selectedOp, aOp, bOp,
                                         ret[linearIdx]);
        }
      }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}

} // namespace mlir::triton::AMD
