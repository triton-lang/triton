#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    Type tgtTy = accum.getType();
    for (auto it = llvm::zip(a, b).begin(); it != llvm::zip(a, b).end(); ++it) {
      const auto &aElem = std::get<0>(*it);
      const auto &bElem = std::get<1>(*it);

      assert(aElem.getType() == tgtTy);
      assert(bElem.getType() == tgtTy);

      // to avoid: 'llvm.intr.fmuladd' op operand #0 must be floating point LLVM
      // type or LLVM dialect-compatible vector of floating point LLVM type, but
      // got 'i32'
      llvm::TypeSwitch<Type>(tgtTy)
          .Case<FloatType>([&](auto) {
            accum = builder.create<LLVM::FMulAddOp>(loc, aElem, bElem, accum);
          })
          .Case<IntegerType>([&](auto) {
            accum = builder.create<LLVM::AddOp>(
                loc, builder.create<LLVM::MulOp>(loc, aElem, bElem), accum);
          });
    }
    return accum;
  }
};

} // namespace

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
