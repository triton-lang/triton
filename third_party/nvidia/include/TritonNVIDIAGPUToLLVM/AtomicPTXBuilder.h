#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_ATOMICPTXBUILDER_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_ATOMICPTXBUILDER_H

#include "PTXAsmFormat.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

namespace mlir::triton::NVIDIA {

enum class PtxAtomicAddrSpace { Global, Shared, SharedCluster };
enum class PtxAtomicInstr { Atom, Red };

inline std::string getPtxRegisterSizeCode(int size, bool isFloat) {
  switch (size) {
  case 1:
    return "b";
  case 16:
    return "h";
  case 32:
    return isFloat ? "f" : "r";
  case 64:
    return isFloat ? "d" : "l";
  case 128:
    return "q";
  default:
    llvm_unreachable("Unsupported register size");
  }
}

inline FailureOr<Value>
emitPtxAtomicRMWImpl(ConversionPatternRewriter &rewriter, Location loc,
                     Type valueElemTy, Value ptr, ArrayRef<Value> vals,
                     RMWOp rmwOpAttr, MemSemantic sem, std::string scope,
                     Value pred, unsigned vec, unsigned packed,
                     PtxAtomicAddrSpace addrSpace, PtxAtomicInstr instr) {
  assert((vec == 1 || packed == 1) && "packed or vec must be 1");
  assert(vals.size() == (vec > 1 ? vec : packed) &&
         "Expected atomic RMW operand count to match vectorization");

  bool isRed = instr == PtxAtomicInstr::Red;
  bool isGlobal = addrSpace == PtxAtomicAddrSpace::Global;
  bool isSharedCluster = addrSpace == PtxAtomicAddrSpace::SharedCluster;
  assert((isGlobal || (vec == 1 && packed == 1)) &&
         "shared atomic RMW does not support vectorized lowering");

  TritonLLVMOpBuilder b(loc, rewriter);
  unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
  Type packedTy = vec_ty(valueElemTy, packed);

  PTXBuilder ptxBuilderAtomicRMW;
  std::string tyId =
      getPtxRegisterSizeCode(valueElemNBits * packed, /*isFloat=*/false);

  PTXBuilder::Operand *dstOpr = nullptr;
  if (!isRed) {
    if (vec > 1) {
      dstOpr = ptxBuilderAtomicRMW.newListOperand();
      for (unsigned ii = 0; ii < vec; ++ii) {
        dstOpr->listAppend(
            ptxBuilderAtomicRMW.newOperand("=" + tyId, /*init=*/true));
      }
    } else {
      dstOpr = ptxBuilderAtomicRMW.newOperand("=" + tyId, /*init=*/true);
    }
  }

  auto *ptrOpr = ptxBuilderAtomicRMW.newAddrOperand(ptr, isGlobal ? "l" : "r");

  PTXBuilder::Operand *valOpr;
  if (vec > 1) {
    valOpr = ptxBuilderAtomicRMW.newListOperand();
    for (Value val : vals)
      valOpr->listAppend(ptxBuilderAtomicRMW.newOperand(val, tyId));
  } else if (packed > 1) {
    Value packedVal = b.undef(packedTy);
    for (auto [idx, val] : llvm::enumerate(vals))
      packedVal = b.insert_element(packedTy, packedVal, val, b.i32_val(idx));
    valOpr = ptxBuilderAtomicRMW.newOperand(packedVal, tyId);
  } else {
    valOpr = ptxBuilderAtomicRMW.newOperand(vals.front(), tyId);
  }

  auto &atomicInstr = *ptxBuilderAtomicRMW.create(isRed ? std::string("red")
                                                        : std::string("atom"));
  if (isGlobal)
    atomicInstr.global();
  else if (isSharedCluster)
    atomicInstr.o("shared::cluster");
  else
    atomicInstr.shared();
  atomicInstr.o(scope);

  std::string rmwOp = stringifyRMWOp(rmwOpAttr).str();
  std::string suffix;
  auto sBits = std::to_string(valueElemNBits);

  switch (rmwOpAttr) {
  case RMWOp::AND:
  case RMWOp::OR:
  case RMWOp::XOR:
  case RMWOp::XCHG:
    suffix = "b" + sBits;
    break;
  case RMWOp::ADD:
    suffix = "u" + sBits;
    break;
  case RMWOp::FADD:
    rmwOp = "add";
    rmwOp += valueElemNBits == 16 ? ".noftz" : "";
    suffix = (valueElemTy.isBF16() ? "bf" : "f") + sBits;
    suffix += packed == 2 && valueElemNBits == 16 ? "x2" : "";
    break;
  case RMWOp::MAX:
  case RMWOp::MIN:
    suffix = "s" + sBits;
    break;
  case RMWOp::UMAX:
    rmwOp = "max";
    suffix = "u" + sBits;
    break;
  case RMWOp::UMIN:
    rmwOp = "min";
    suffix = "u" + sBits;
    break;
  default:
    return failure();
  }
  if (isRed && rmwOpAttr == RMWOp::XCHG)
    return failure();

  std::string semStr;
  llvm::raw_string_ostream os(semStr);
  os << sem;
  atomicInstr.o(semStr).o(rmwOp).v(vec).o(suffix);
  if (isRed) {
    atomicInstr(ptrOpr, valOpr).maybePredicate(pred);
    return ptxBuilderAtomicRMW.launch(rewriter, loc,
                                      void_ty(rewriter.getContext()));
  }

  atomicInstr(dstOpr, ptrOpr, valOpr).maybePredicate(pred);

  Type retType;
  if (vec > 1) {
    SmallVector<Type> retTys(vec, valueElemTy);
    retType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), retTys);
  } else if (packed > 1) {
    retType = packedTy;
  } else {
    retType = valueElemTy;
  }
  return ptxBuilderAtomicRMW.launch(rewriter, loc, retType);
}

inline Value emitLLVMAtomicRMW(ConversionPatternRewriter &rewriter,
                               Location loc, Value ptr, Value val,
                               LLVM::AtomicBinOp binOp,
                               LLVM::AtomicOrdering ordering,
                               StringRef syncScope) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Type valueTy = val.getType();
  // Exchange uses the bit pattern even when the source value is floating point.
  bool bitcast = binOp == LLVM::AtomicBinOp::xchg && isa<FloatType>(valueTy);
  if (bitcast)
    val = b.bitcast(val, int_ty(valueTy.getIntOrFloatBitWidth()));
  Value old = LLVM::AtomicRMWOp::create(rewriter, loc, binOp, ptr, val,
                                        ordering, syncScope);
  return bitcast ? b.bitcast(old, valueTy) : old;
}

inline FailureOr<Value>
emitAtomicRMW(ConversionPatternRewriter &rewriter, Location loc,
              Type valueElemTy, Value ptr, ArrayRef<Value> vals,
              RMWOp rmwOpAttr, MemSemantic sem, MemSyncScope scope, Value pred,
              const TargetInfoBase &targetInfo, unsigned vec = 1,
              unsigned packed = 1, bool returnOld = true) {
  if (pred || vec != 1 || packed != 1)
    return emitPtxAtomicRMWImpl(
        rewriter, loc, valueElemTy, ptr, vals, rmwOpAttr, sem,
        stringifyMemSyncScope(scope).str(), pred, vec, packed,
        PtxAtomicAddrSpace::Global,
        returnOld ? PtxAtomicInstr::Atom : PtxAtomicInstr::Red);
  auto binOp = matchAtomicOp(rmwOpAttr);
  auto ordering = getMemoryOrdering(sem);
  if (!binOp || !ordering)
    return failure();
  return emitLLVMAtomicRMW(rewriter, loc, ptr, vals.front(), *binOp, *ordering,
                           targetInfo.getAtomicSyncScope(scope));
}

inline FailureOr<Value>
emitPtxSharedAtomicRMW(ConversionPatternRewriter &rewriter, Location loc,
                       Type valueElemTy, Value ptr, ArrayRef<Value> vals,
                       RMWOp rmwOpAttr, Value pred, bool isCluster,
                       PtxAtomicInstr instr) {
  if (!pred) {
    auto binOp = matchAtomicOp(rmwOpAttr);
    if (!binOp || (instr == PtxAtomicInstr::Red && rmwOpAttr == RMWOp::XCHG))
      return failure();
    return emitLLVMAtomicRMW(rewriter, loc, ptr, vals.front(), *binOp,
                             LLVM::AtomicOrdering::monotonic,
                             isCluster ? "cluster" : "block");
  }
  auto addrSpace = isCluster ? PtxAtomicAddrSpace::SharedCluster
                             : PtxAtomicAddrSpace::Shared;
  return emitPtxAtomicRMWImpl(rewriter, loc, valueElemTy, ptr, vals, rmwOpAttr,
                              MemSemantic::RELAXED,
                              isCluster ? "cluster" : "cta", pred, /*vec=*/1,
                              /*packed=*/1, addrSpace, instr);
}

inline Value emitPtxAtomicCAS(ConversionPatternRewriter &rewriter, Location loc,
                              Type valueElemTy, Value ptr, Value cmp, Value val,
                              MemSemantic sem, MemSyncScope scope, Value pred) {
  unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
  PTXBuilder ptxBuilderAtomicCAS;
  std::string tyId = getPtxRegisterSizeCode(valueElemNBits, /*isFloat=*/false);
  auto *dstOpr = ptxBuilderAtomicCAS.newOperand("=" + tyId, /*init=*/true);
  auto *ptrOpr = ptxBuilderAtomicCAS.newAddrOperand(ptr, "l");
  auto *cmpOpr = ptxBuilderAtomicCAS.newOperand(cmp, tyId);
  auto *valOpr = ptxBuilderAtomicCAS.newOperand(val, tyId);
  auto &atom = *ptxBuilderAtomicCAS.create("atom");
  auto sTy = "b" + std::to_string(valueElemNBits);
  std::string semStr;
  llvm::raw_string_ostream os(semStr);
  os << sem;
  atom.global().o(semStr).o(stringifyMemSyncScope(scope).str()).o("cas").o(sTy);
  atom(dstOpr, ptrOpr, cmpOpr, valOpr).maybePredicate(pred);
  return ptxBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
}

inline Value emitAtomicCAS(ConversionPatternRewriter &rewriter, Location loc,
                           Type valueElemTy, Value ptr, Value cmp, Value val,
                           MemSemantic sem, MemSyncScope scope, Value pred,
                           const TargetInfoBase &targetInfo) {
  // LLVM widens 16-bit cmpxchg to a 32-bit CAS, touching adjacent storage.
  if (pred || valueElemTy.getIntOrFloatBitWidth() == 16)
    return emitPtxAtomicCAS(rewriter, loc, valueElemTy, ptr, cmp, val, sem,
                            scope, pred);
  TritonLLVMOpBuilder b(loc, rewriter);
  bool bitcast = isa<FloatType>(valueElemTy);
  if (bitcast) {
    auto intTy = int_ty(valueElemTy.getIntOrFloatBitWidth());
    cmp = b.bitcast(cmp, intTy);
    val = b.bitcast(val, intTy);
  }
  auto ordering = *getMemoryOrdering(sem);
  // PTX acquire also applies when the comparison fails.
  auto failureOrdering =
      sem == MemSemantic::ACQUIRE || sem == MemSemantic::ACQUIRE_RELEASE
          ? LLVM::AtomicOrdering::acquire
          : LLVM::AtomicOrdering::monotonic;
  Value result = LLVM::AtomicCmpXchgOp::create(
      rewriter, loc, ptr, cmp, val, ordering, failureOrdering,
      targetInfo.getAtomicSyncScope(scope));
  Value old = b.extract_val(result, 0);
  return bitcast ? b.bitcast(old, valueElemTy) : old;
}

} // namespace mlir::triton::NVIDIA

#endif // TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_ATOMICPTXBUILDER_H
