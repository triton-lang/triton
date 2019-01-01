#ifndef TDL_INCLUDE_IR_BUILDER_H
#define TDL_INCLUDE_IR_BUILDER_H

#include <vector>
#include <string>

namespace tdl{
namespace ir{

class basic_block;
class value;
class type;
class constant_int;

/* Builder */
class builder{
public:
  void set_insert_point(basic_block* bb);
  basic_block* get_insert_block();
  // Constants
  value *get_int32(unsigned val);
  // Types
  type *get_float_ty();
  type *get_double_ty();
  // Branch instructions
  value* create_br(basic_block *bb);
  value* create_cond_br(value *cond, basic_block* if_bb, basic_block* else_bb);
  // Cast instructions
  value* create_si_to_fp(value *src, type *dst_ty);
  value* create_ui_to_fp(value *src, type *dst_ty);
  value* create_fp_to_si(value *src, type *dst_ty);
  value* create_fp_to_ui(value *src, type *dst_ty);
  value* create_fp_ext(value *src, type *dst_ty);
  value* create_fp_trunc(value *src, type *dst_ty);
  value* create_int_cast(value *src, type *dst_ty, bool is_signed);
  // Call instruction
  value* create_call(value *fn, const std::vector<value*> &args);
  // Binary instructions
  value *create_fmul(value *lhs, value *rhs, const std::string &name = "");
  value *create_mul(value *lhs, value *rhs, const std::string &name = "");
  value *create_fdiv(value *lhs, value *rhs, const std::string &name = "");
  value *create_sdiv(value *lhs, value *rhs, const std::string &name = "");
  value *create_udiv(value *lhs, value *rhs, const std::string &name = "");
  value *create_frem(value *lhs, value *rhs, const std::string &name = "");
  value *create_srem(value *lhs, value *rhs, const std::string &name = "");
  value *create_urem(value *lhs, value *rhs, const std::string &name = "");
  value *create_fadd(value *lhs, value *rhs, const std::string &name = "");
  value *create_add(value *lhs, value *rhs, const std::string &name = "");
  value *create_gep(value *lhs, const std::vector<value*> &offs, const std::string &name = "");
  value *create_fsub(value *lhs, value *rhs, const std::string &name = "");
  value *create_sub(value *lhs, value *rhs, const std::string &name = "");
  value *create_lshr(value *lhs, value *rhs, const std::string &name = "");
  value *create_ashr(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOLT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSLT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpULT(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOGT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSGT(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpUGT(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOLE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSLE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpULE(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOGE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpSGE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpUGE(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpOEQ(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpEQ(value *lhs, value *rhs, const std::string &name = "");
  value *create_fcmpONE(value *lhs, value *rhs, const std::string &name = "");
  value *create_icmpNE(value *lhs, value *rhs, const std::string &name = "");
  value *create_and(value *lhs, value *rhs, const std::string &name = "");
  value *create_xor(value *lhs, value *rhs, const std::string &name = "");
  value *create_or(value *lhs, value *rhs, const std::string &name = "");
  // Side effects
  value *create_fneg(value *arg, const std::string &name = "");
  value *create_neg(value *arg, const std::string &name = "");
  value *create_load(value *arg, const std::string &name = "");
  value *create_not(value *arg, const std::string &name = "");
  // Tile instruction
  value *create_splat(value *arg, const std::vector<unsigned> &shapes, const std::string &name = "");
  value *create_reshape(value *arg, const std::vector<unsigned> &shapes, const std::string &name = "");
  value *create_broadcast(value *arg, const std::vector<unsigned> &shapes, const std::string &name = "");
  // Terminators
  value *create_ret_void();
};

}
}

#endif
