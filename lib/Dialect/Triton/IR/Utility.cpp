#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;

Value tt::getPredMask(RewriterBase &rewriter, Type typeLike, Value currentMask,
                      Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType)) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}
