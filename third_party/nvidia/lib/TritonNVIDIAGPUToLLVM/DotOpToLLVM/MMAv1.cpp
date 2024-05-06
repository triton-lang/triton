#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

static Type getMmaRetType(TensorType operand) {
  auto *ctx = operand.getContext();
  Type fp32Ty = type::f32Ty(ctx);
  // f16*f16+f32->f32
  return struct_ty(SmallVector<Type>{8, fp32Ty});
}

static ValueTable extractLoadedOperand(Value llStruct, int NK,
                                       ConversionPatternRewriter &rewriter,
                                       const LLVMTypeConverter *typeConverter,
                                       Type type) {
  ValueTable rcds;
  SmallVector<Value> elems =
      unpackLLElements(llStruct.getLoc(), llStruct, rewriter);

  int offset = 0;
  for (int i = 0; offset < elems.size(); ++i) {
    for (int k = 0; k < NK; k += 4) {
      rcds[{i, k}] = std::make_pair(elems[offset], elems[offset + 1]);
      offset += 2;
    }
  }

  return rcds;
}

LogicalResult convertMMA884(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = op.getContext();
  auto loc = op.getLoc();

  Value A = op.getA();
  Value B = op.getB();
  Value D = op.getResult();
  auto mmaLayout = cast<NvidiaMmaEncodingAttr>(
      cast<RankedTensorType>(D.getType()).getEncoding());
  auto ALayout = cast<DotOperandEncodingAttr>(
      cast<RankedTensorType>(A.getType()).getEncoding());
  auto BLayout = cast<DotOperandEncodingAttr>(
      cast<RankedTensorType>(B.getType()).getEncoding());

  auto ATensorTy = cast<RankedTensorType>(A.getType());
  auto BTensorTy = cast<RankedTensorType>(B.getType());
  auto DTensorTy = cast<RankedTensorType>(D.getType());
  auto AShape = ATensorTy.getShape();
  auto BShape = BTensorTy.getShape();

  bool isARow = mmaLayout.getMMAv1IsRow(ALayout.getOpIdx());
  bool isBRow = mmaLayout.getMMAv1IsRow(BLayout.getOpIdx());
  auto [isARow_, isBRow_, isAVec4_, isBVec4_, _] =
      mmaLayout.decodeVoltaLayoutStates();
  assert(isARow == isARow_);
  assert(isBRow == isBRow_);

  unsigned numM = mmaLayout.getMMAv1NumOuter(AShape, ALayout.getOpIdx());
  unsigned numN = mmaLayout.getMMAv1NumOuter(BShape, BLayout.getOpIdx());
  unsigned NK = AShape[1];

  auto has = extractLoadedOperand(adaptor.getA(), NK, rewriter, typeConverter,
                                  ATensorTy);
  auto hbs = extractLoadedOperand(adaptor.getB(), NK, rewriter, typeConverter,
                                  BTensorTy);

  // Initialize accumulators with external values, the acc holds the
  // accumulator value that is shared between the MMA instructions inside a
  // DotOp, we can call the order of the values the accumulator-internal
  // order.
  SmallVector<Value> acc = unpackLLElements(loc, adaptor.getC(), rewriter);
  size_t resSize = acc.size();

  // The resVals holds the final result of the DotOp.
  // NOTE The current order of resVals is different from acc, we call it the
  // accumulator-external order. and
  SmallVector<Value> resVals(resSize);

  auto getIdx = [&](int m, int n) {
    std::vector<size_t> idx{{
        (m * 2 + 0) + (n * 4 + 0) * numM, // row0
        (m * 2 + 0) + (n * 4 + 1) * numM,
        (m * 2 + 1) + (n * 4 + 0) * numM, // row1
        (m * 2 + 1) + (n * 4 + 1) * numM,
        (m * 2 + 0) + (n * 4 + 2) * numM, // row2
        (m * 2 + 0) + (n * 4 + 3) * numM,
        (m * 2 + 1) + (n * 4 + 2) * numM, // row3
        (m * 2 + 1) + (n * 4 + 3) * numM,
    }};
    return idx;
  };

  auto callMMA = [&](unsigned m, unsigned n, unsigned k) {
    auto ha = has.at({m, k});
    auto hb = hbs.at({n, k});

    PTXBuilder builder;
    auto idx = getIdx(m, n);

    // note: using "=f" for float leads to cleaner PTX
    bool isIntMMA = DTensorTy.getElementType().isInteger(32);
    auto *resOprs = builder.newListOperand(8, isIntMMA ? "=r" : "=f");
    auto *AOprs = builder.newListOperand({
        {ha.first, "r"},
        {ha.second, "r"},
    });

    auto *BOprs = builder.newListOperand({
        {hb.first, "r"},
        {hb.second, "r"},
    });
    auto *COprs = builder.newListOperand();
    for (int i = 0; i < 8; ++i)
      COprs->listAppend(builder.newOperand(acc[idx[i]], std::to_string(i)));

    auto mma = builder.create("mma.sync.aligned.m8n8k4")
                   ->o(isARow ? "row" : "col")
                   .o(isBRow ? "row" : "col")
                   .o("f32.f16.f16.f32");

    mma(resOprs, AOprs, BOprs, COprs);

    Value res = builder.launch(rewriter, loc, getMmaRetType(ATensorTy));

    for (auto i = 0; i < 8; i++) {
      Value elem = extract_val(f32_ty, res, i);
      acc[idx[i]] = elem;
    }
  };

  for (unsigned k = 0; k < NK; k += 4)
    for (unsigned m = 0; m < numM / 2; ++m)
      for (unsigned n = 0; n < numN / 2; ++n) {
        callMMA(m, n, k);
      }

  // res holds the same layout of acc
  for (size_t i = 0; i < acc.size(); ++i) {
    resVals[i] = acc[i];
  }

  Value res = packLLElements(loc, typeConverter, resVals, rewriter, DTensorTy);
  rewriter.replaceOp(op, res);
  return success();
}
