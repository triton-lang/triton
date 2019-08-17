#ifndef TRITON_IR_ENUMS_H
#define TRITON_IR_ENUMS_H

namespace triton{
namespace ir{


enum binary_op_t {
  Add,
  FAdd,
  Sub,
  FSub,
  Mul,
  FMul,
  UDiv,
  SDiv,
  FDiv,
  URem,
  SRem,
  FRem,
  Shl,
  LShr,
  AShr,
  And,
  Or,
  Xor
};

enum cast_op_t {
  Trunc,
  ZExt,
  SExt,
  FPTrunc,
  FPExt,
  UIToFP,
  SIToFP,
  FPToUI,
  FPToSI,
  PtrToInt,
  IntToPtr,
  BitCast,
  AddrSpaceCast
};

enum cmp_pred_t {
  FIRST_FCMP_PREDICATE,
  FCMP_FALSE,
  FCMP_OEQ,
  FCMP_OGT,
  FCMP_OGE,
  FCMP_OLT,
  FCMP_OLE,
  FCMP_ONE,
  FCMP_ORD,
  FCMP_UNO,
  FCMP_UEQ,
  FCMP_UGT,
  FCMP_UGE,
  FCMP_ULT,
  FCMP_ULE,
  FCMP_UNE,
  FCMP_TRUE,
  LAST_FCMP_PREDICATE,
  FIRST_ICMP_PREDICATE,
  ICMP_EQ,
  ICMP_NE,
  ICMP_UGT,
  ICMP_UGE,
  ICMP_ULT,
  ICMP_ULE,
  ICMP_SGT,
  ICMP_SGE,
  ICMP_SLT,
  ICMP_SLE,
  LAST_ICMP_PREDICATE
};




}
}

#endif
