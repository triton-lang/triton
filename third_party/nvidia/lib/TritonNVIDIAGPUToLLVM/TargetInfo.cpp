#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
namespace mlir::triton::NVIDIA {
// Check if the reduction can use a redux op and return the kind.
static std::optional<NVVM::ReduxKind> matchReduxKind(triton::ReduceOp op,
                                                     int computeCapability) {
  if (computeCapability < 80)
    return std::nullopt;
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return std::nullopt;
  Block *block = &(*op.getCombineOp().begin());
  Operation *yield = block->getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return std::nullopt;
  auto intType = reduceOp->getResultTypes()[0].dyn_cast<IntegerType>();
  if (!intType || intType.getWidth() > 32)
    return std::nullopt;
  if (reduceOp->getOperand(0) != block->getArgument(0) ||
      reduceOp->getOperand(1) != block->getArgument(1))
    return std::nullopt;
  if (isa<arith::AddIOp>(reduceOp))
    return NVVM::ReduxKind::ADD;
  if (isa<arith::AndIOp>(reduceOp))
    return NVVM::ReduxKind::AND;
  if (isa<arith::OrIOp>(reduceOp))
    return NVVM::ReduxKind::OR;
  if (isa<arith::XOrIOp>(reduceOp))
    return NVVM::ReduxKind::XOR;
  if (isa<arith::MinSIOp>(reduceOp))
    return NVVM::ReduxKind::MIN;
  if (isa<arith::MinUIOp>(reduceOp))
    return NVVM::ReduxKind::UMIN;
  if (isa<arith::MaxSIOp>(reduceOp))
    return NVVM::ReduxKind::MAX;
  if (isa<arith::MaxUIOp>(reduceOp))
    return NVVM::ReduxKind::UMAX;
  return std::nullopt;
}

bool TargetInfo::supportMaximumMinimum() const {
  return computeCapability >= 80;
}
Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  Value threadMask = int_val(type.getIntOrFloatBitWidth(), -1);
  return rewriter.create<NVVM::VoteBallotOp>(loc, type, threadMask, cmp);
}
Value TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                              Value ptr, Value val, Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  unsigned bits = std::max(8u, val.getType().getIntOrFloatBitWidth());
  const char *c = bits == 64 ? "l" : (bits == 16 ? "h" : "r");

  PTXBuilder builder;
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto *valOpr = builder.newOperand(val, c);
  auto &st = builder.create<>("st")->shared().b(bits);
  st(ptrOpr, valOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, void_ty(ctx));
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Type elemTy, Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = ptr.getType().cast<LLVM::LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  PTXBuilder builder;
  auto *dOpr = builder.newOperand(c);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  ld(dOpr, ptrOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, elemTy);
}

Value TargetInfo::shuffleXor(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return LLVM::NVIDIA::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, int i) const {
  return LLVM::NVIDIA::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, Value i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  if (auto kind = matchReduxKind(op, computeCapability)) {
    // Based on benchmarking on A100 redux op gives a speed up only when doing
    // a single reduction (not partitioned) and when the mask is static.
    // Therefore we currently only enable it to reduce across all the lanes.
    if (numLaneToReduce == 32) {
      assert(acc.size() == 1);
      Value mask = i32_val(0xFFFFFFFF);
      // Even though we currently don't use redux for partitioned reduction
      // the code below supports it in case we want to tweak the heuristic.
      if (numLaneToReduce < 32) {
        // For partitioned reduction we need to calculate the mask so that
        // each group of numLaneToReduce threads has the correct mask.
        unsigned bitmask = (1 << numLaneToReduce) - 1;
        Value threadId = getThreadId(rewriter, loc);
        Value laneId = urem(threadId, i32_val(32));
        mask = shl(i32_val(bitmask),
                   and_(laneId, i32_val(~(numLaneToReduce - 1))));
      }
      for (unsigned i = 0; i < acc.size(); ++i) {
        unsigned bitwidth = acc[i].getType().cast<IntegerType>().getWidth();
        if (bitwidth < 32) {
          if (*kind == NVVM::ReduxKind::MIN || *kind == NVVM::ReduxKind::MAX)
            acc[i] = sext(i32_ty, acc[i]);
          else
            acc[i] = zext(i32_ty, acc[i]);
        }
        acc[i] = rewriter.create<NVVM::ReduxOp>(loc, acc[i].getType(), acc[0],
                                                *kind, mask);
        if (bitwidth < 32)
          acc[i] = trunc(int_ty(bitwidth), acc[i]);
      }
      return true;
    }
  }
  return false;
}
} // namespace mlir::triton::NVIDIA
