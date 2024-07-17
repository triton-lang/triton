#ifndef TRITONCPU_CONVERSION_TRITONCPUOPT_OPTCOMMON_H
#define TRITONCPU_CONVERSION_TRITONCPUOPT_OPTCOMMON_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace triton {
namespace cpu {

inline Type getElemTyOrTy(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.getElementType();
  return ty;
}

inline bool isTyOrVectorOf(Type ty, Type elemTy) {
  return getElemTyOrTy(ty) == elemTy;
}

inline bool isBf16(Type ty) {
  return isTyOrVectorOf(ty, BFloat16Type::get(ty.getContext()));
}

inline bool isFp16(Type ty) {
  return isTyOrVectorOf(ty, Float16Type::get(ty.getContext()));
}

inline bool isFp32(Type ty) {
  return isTyOrVectorOf(ty, Float32Type::get(ty.getContext()));
}

inline bool isFp8(Type ty) {
  Type elemTy = getElemTyOrTy(ty);
  if (elemTy.isIntOrFloat() && !elemTy.isInteger())
    return elemTy.getIntOrFloatBitWidth() == 8;
  return false;
}

inline Type toTyOrVectorOf(Type ty, Type elemTy) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.cloneWith(std::nullopt, elemTy);
  return elemTy;
}

inline Type toInt8(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 8));
}

inline Type toInt16(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 16));
}

inline Type toInt32(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 32));
}

inline Type toFp8E5M2(Type ty) {
  return toTyOrVectorOf(ty, Float8E5M2Type::get(ty.getContext()));
}

inline Type toFp16(Type ty) {
  return toTyOrVectorOf(ty, Float16Type::get(ty.getContext()));
}

inline Type toBf16(Type ty) {
  return toTyOrVectorOf(ty, BFloat16Type::get(ty.getContext()));
}

inline Type toFp32(Type ty) {
  return toTyOrVectorOf(ty, Float32Type::get(ty.getContext()));
}

inline Value intCst(Location loc, Type ty, int64_t val,
                    PatternRewriter &rewriter) {
  TypedAttr valAttr = IntegerAttr::get(getElemTyOrTy(ty), val);
  if (auto vecTy = dyn_cast<VectorType>(ty))
    valAttr = SplatElementsAttr::get(vecTy, valAttr);
  return rewriter.create<arith::ConstantOp>(loc, valAttr);
}

inline Value fpCst(Location loc, Type ty, double val,
                   PatternRewriter &rewriter) {
  TypedAttr valAttr = FloatAttr::get(getElemTyOrTy(ty), val);
  if (auto vecTy = dyn_cast<VectorType>(ty))
    valAttr = SplatElementsAttr::get(vecTy, valAttr);
  return rewriter.create<arith::ConstantOp>(loc, valAttr);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
Value cstLike(Location loc, Value tySrc, T val, PatternRewriter &rewriter) {
  return intCst(loc, tySrc.getType(), val, rewriter);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
Value cstLike(Location loc, Value tySrc, T val, PatternRewriter &rewriter) {
  return fpCst(loc, tySrc.getType(), val, rewriter);
}

} // namespace cpu
} // namespace triton
} // namespace mlir

#define int_cst(ty, val) intCst(loc, ty, val, rewriter)
#define cst_like(src, val) cstLike(loc, src, val, rewriter)

#define op_addi(lhs, rhs) rewriter.create<arith::AddIOp>(loc, lhs, rhs)
#define op_addf(lhs, rhs) rewriter.create<arith::AddFOp>(loc, lhs, rhs)
#define op_subi(lhs, rhs) rewriter.create<arith::SubIOp>(loc, lhs, rhs)
#define op_subf(lhs, rhs) rewriter.create<arith::SubFOp>(loc, lhs, rhs)
#define op_mulf(lhs, rhs) rewriter.create<arith::MulFOp>(loc, lhs, rhs)
#define op_bitcast(ty, val) rewriter.create<arith::BitcastOp>(loc, ty, val)
#define op_lshr(lhs, rhs) rewriter.create<arith::ShRUIOp>(loc, lhs, rhs)
#define op_shl(lhs, rhs) rewriter.create<arith::ShLIOp>(loc, lhs, rhs)
#define op_trunci(ty, val) rewriter.create<arith::TruncIOp>(loc, ty, val)
#define op_zext(ty, val) rewriter.create<arith::ExtUIOp>(loc, ty, val)
#define op_sext(ty, val) rewriter.create<arith::ExtSIOp>(loc, ty, val)
#define op_and(lhs, rhs) rewriter.create<arith::AndIOp>(loc, lhs, rhs)
#define op_or(lhs, rhs) rewriter.create<arith::OrIOp>(loc, lhs, rhs)
#define op_minui(lhs, rhs) rewriter.create<arith::MinUIOp>(loc, lhs, rhs)
#define op_maxui(lhs, rhs) rewriter.create<arith::MaxUIOp>(loc, lhs, rhs)
#define op_select(cond, val, other)                                            \
  rewriter.create<arith::SelectOp>(loc, cond, val, other)
#define op_sitofp(ty, val) rewriter.create<arith::SIToFPOp>(loc, ty, val)
#define op_fptosi(ty, val) rewriter.create<arith::FPToSIOp>(loc, ty, val)

#define op_icmp_eq(lhs, rhs)                                                   \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs)
#define op_icmp_ne(lhs, rhs)                                                   \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs)
#define op_icmp_ugt(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs, rhs)
#define op_icmp_uge(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, lhs, rhs)
#define op_icmp_ult(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lhs, rhs)
#define op_icmp_ule(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, lhs, rhs)
#define op_icmp_sgt(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs)
#define op_icmp_sge(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs)
#define op_icmp_slt(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs)
#define op_icmp_sle(lhs, rhs)                                                  \
  rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs, rhs)

#endif
