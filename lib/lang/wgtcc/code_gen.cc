#include "triton/lang/wgtcc/code_gen.h"
#include "triton/lang/wgtcc/evaluator.h"
#include "triton/lang/wgtcc/parser.h"
#include "triton/lang/wgtcc/token.h"
#include "triton/ir/module.h"

// Helpers
void Generator::set_ret(ir::value* value) {
  ret_ = value;
}

inline bool is_terminator(ir::value* x) {
  return x && dynamic_cast<ir::terminator_inst*>(x);
}


// Expression

void Generator::VisitBinaryOp(BinaryOp* binary) {
  Visit(binary->rhs_);
  ir::value* rhs = ret_;

  if(binary->op_ == '=')
    return set_ret(assign_->GenExpr(binary->lhs_, rhs));

  Visit(binary->lhs_);
  ir::value* lhs = ret_;
  // op info
  auto type = binary->lhs_->Type();
  auto flt = type->IsFloat();
  auto sign = !type->IsUnsigned();
  // return
  switch(binary->op_){
    case Token::LOGICAL_AND: return set_ret(bld_->create_and(lhs, rhs));
    case Token::LOGICAL_OR: return set_ret(bld_->create_or(lhs, rhs));
    case '|': return set_ret(bld_->create_or(lhs, rhs));
    case '&': return set_ret(bld_->create_and(lhs, rhs));
    case '^': return set_ret(bld_->create_xor(lhs, rhs));
    case Token::LEFT: return set_ret(bld_->create_shl(lhs, rhs));
    case Token::RIGHT: return set_ret(bld_->create_lshr(lhs, rhs));
    case '.': return error_not_implemented();
    case ',': return error_not_implemented();
    case '+':
      if(binary->lhs_->Type()->ToPointer())
        return set_ret(bld_->create_gep(lhs, {rhs}));
      else if(flt)
        return set_ret(bld_->create_fadd(lhs, rhs));
      else
        return set_ret(bld_->create_add(lhs, rhs));
    case '-':
      if(binary->lhs_->Type()->ToPointer())
        return set_ret(bld_->create_gep(lhs, {bld_->create_neg(rhs)}));
      else if(flt)
        return set_ret(bld_->create_fsub(lhs, rhs));
      else
        return set_ret(bld_->create_sub(lhs, rhs));
    case '*':
      if(flt)
        return set_ret(bld_->create_fmul(lhs, rhs));
      else
        return set_ret(bld_->create_mul(lhs, rhs));
    case '/':
      if(flt)
        return set_ret(bld_->create_fdiv(lhs, rhs));
      else if(sign)
        return set_ret(bld_->create_sdiv(lhs, rhs));
      else if(!sign)
        return set_ret(bld_->create_udiv(lhs, rhs));
      else
        return should_not_happen();
    case '%':
      if(flt)
        return set_ret(bld_->create_frem(lhs, rhs));
      else if(sign)
        return set_ret(bld_->create_srem(lhs, rhs));
      else
        return set_ret(bld_->create_urem(lhs, rhs));
    case '<':
      if(flt)
        return set_ret(bld_->create_fcmpOLT(lhs, rhs));
      else if(sign)
        return set_ret(bld_->create_icmpSLT(lhs, rhs));
      else if(!sign)
        return set_ret(bld_->create_icmpULT(lhs, rhs));
      else
        return should_not_happen();
    case '>':
      if(flt)
        return set_ret(bld_->create_fcmpOGT(lhs, rhs));
      else if(sign)
        return set_ret(bld_->create_icmpSGT(lhs, rhs));
      else if(!sign)
        return set_ret(bld_->create_icmpUGT(lhs, rhs));
      else
        return should_not_happen();
    case Token::LE:
      if(flt)
        return set_ret(bld_->create_fcmpOLE(lhs, rhs));
      else if(sign)
        return set_ret(bld_->create_icmpSLE(lhs, rhs));
      else if(!sign)
        return set_ret(bld_->create_icmpULE(lhs, rhs));
      else
        return should_not_happen();
    case Token::GE:
      if(flt)
        return set_ret(bld_->create_fcmpOGE(lhs, rhs));
      else if(sign)
        return set_ret(bld_->create_icmpSGE(lhs, rhs));
      else if(!sign)
        return set_ret(bld_->create_icmpUGE(lhs, rhs));
      else
        return should_not_happen();
    case Token::EQ:
      if(flt)
        return set_ret(bld_->create_fcmpOEQ(lhs, rhs));
      else
        return set_ret(bld_->create_icmpEQ(lhs, rhs));
    case Token::NE:
      if(flt)
        return set_ret(bld_->create_fcmpONE(lhs, rhs));
      else
        return set_ret(bld_->create_icmpEQ(lhs, rhs));
    default:
      error_not_implemented();
  }
  error_not_implemented();
}

void Generator::VisitUnaryOp(UnaryOp* unary) {
  // recursion
  Visit(unary->operand_);
  ir::value* op = ret_;
  ir::type* type = GenIRType(unary->operand_->Type(), *ctx_);
  // return
  switch  (unary->op_) {
    case Token::PREFIX_INC: return error_not_implemented();
    case Token::PREFIX_DEC: return error_not_implemented();
    case Token::POSTFIX_INC: return error_not_implemented();
    case Token::POSTFIX_DEC: return error_not_implemented();
    case Token::ADDR: return error_not_implemented();
    case Token::DEREF: return error_not_implemented();
    case Token::PLUS: return error_not_implemented();
    case Token::MINUS: return error_not_implemented();
    case '~': return set_ret(bld_->create_neg(op));
    case '!': return set_ret(bld_->create_not(op));
    case Token::CAST: return set_ret(GenCastOp(op, type));
    default: assert(false);
  }
  return error_not_implemented();
}

void Generator::VisitConditionalOp(ConditionalOp* condOp) {
  return error_not_implemented();
}

void Generator::VisitFuncCall(FuncCall* funcCall) {
  std::string name = funcCall->Name();
  if(name == "get_program_id"){
    VisitExpr(funcCall->Args()->at(0));
    ir::value* ret = ret_;
    if(auto axis = dynamic_cast<ir::constant_int*>(ret))
      return set_ret(bld_->create_get_program_id(axis->get_value()));
  }
  return error_not_implemented();
}

void Generator::VisitObject(Object* obj) {
  return error_not_implemented();
}

void Generator::VisitEnumerator(Enumerator* enumer) {
  return error_not_implemented();
}

void Generator::VisitIdentifier(Identifier* ident) {
  return set_ret(mod_->get_value(ident->Name()));
}

void Generator::VisitConstant(Constant* cons) {
  Type* ctype = cons->Type();
  ir::type *type = GenIRType(cons->Type(), *ctx_);
  if(ctype->IsInteger())
    return set_ret(ir::constant_int::get(type, cons->IVal()));
  if(ctype->IsFloat() && ctype->IsReal())
    return set_ret(ir::constant_fp::get(type, cons->FVal()));
  return error_not_implemented();
}

void Generator::VisitTempVar(TempVar* tempVar) {
  return error_not_implemented();
}

// Statement
void Generator::VisitDeclaration(Declaration* decl) {
  auto obj = decl->obj_;
  // initialize to undef
  ir::type* ty = GenIRType(obj->Type(), *ctx_);
  ir::value* val = ir::undef_value::get(ty);
  // compute initializers
  std::vector<ir::value*> inits;
  for (const Initializer& init: decl->Inits()) {
    VisitExpr(init.expr_);
    inits.push_back(ret_);
  }
  // initialize declaration
  ir::type::id_t id = ty->get_type_id();
  if(id == ir::type::StructTyID)
    assert(false);
  if(inits.size() > 1)
    assert(false);
  val = inits[0];
  assert(val->get_type() == ty);
  // update scope symbols table
  const std::string &name = obj->Name();
  if(!name.empty()){
    mod_->set_value(name, val);
    mod_->get_scope().types[name] = ty;
  }
}

void Generator::VisitEmptyStmt(EmptyStmt*) {
  return;
}

void Generator::VisitIfStmt(IfStmt* ifStmt) {
  ir::function *fn = bld_->get_insert_block()->get_parent();
  Stmt *then_ = ifStmt->then_;
  Stmt *else_ = ifStmt->else_;
  VisitExpr(ifStmt->cond_);
  ir::value* cond = ret_;
  ir::basic_block *then_bb = ir::basic_block::create(*ctx_, "then", fn);
  ir::basic_block *else_bb = else_? ir::basic_block::create(*ctx_, "else", fn) : nullptr;
  ir::basic_block *endif_bb = ir::basic_block::create(*ctx_, "endif", fn);
  // seal blocks
  mod_->seal_block(then_bb);
  if(else_bb)
    mod_->seal_block(else_bb);
  // branches
  if(else_)
    bld_->create_cond_br(cond, then_bb, else_bb);
  else
    bld_->create_cond_br(cond, then_bb, endif_bb);
  // then
  bld_->set_insert_point(then_bb);
  VisitStmt(then_);
  if(!is_terminator(ret_))
    bld_->create_br(endif_bb);
  // else
  if(else_){
    bld_->set_insert_point(else_bb);
    VisitStmt(else_);
    if(!is_terminator(ret_))
      bld_->create_br(endif_bb);
  }
  // endif
  mod_->seal_block(endif_bb);
  bld_->set_insert_point(endif_bb);
}

void Generator::VisitJumpStmt(JumpStmt* jumpStmt) {
  return error_not_implemented();
}

void Generator::VisitReturnStmt(ReturnStmt* returnStmt) {
  ir::value *ret;
  if(returnStmt->expr_)
    return error_not_implemented();
  else
    ret = bld_->create_ret_void();
  return set_ret(ret);
}

void Generator::VisitLabelStmt(LabelStmt* labelStmt) {
  return error_not_implemented();
}

void Generator::VisitCompoundStmt(CompoundStmt* compoundStmt) {
  if (compoundStmt->scope_){
    AllocObjects(compoundStmt->scope_);
    pushScope();
  }
  for (auto stmt: compoundStmt->stmts_)
    Visit(stmt);
  if(compoundStmt->scope_)
    popScope();
}

void Generator::VisitFuncDef(FuncDef* funcDef) {
  return error_not_implemented();
}

void Generator::VisitTranslationUnit(TranslationUnit* unit) {
  for (auto extDecl: unit->ExtDecls())
    Visit(extDecl);
}

void Generator::Gen(ir::module *mod) {
  pushScope();
  mod_ = mod;
  ctx_ = &mod_->get_context();
  bld_ = &mod_->get_builder();
  std::unique_ptr<LValAssigner> assign(new LValAssigner(this));
  assign_ = assign.get();
  VisitTranslationUnit(parser_->Unit());
  assign_ = nullptr;
}


// Triton-IR Values

ir::value* Generator::GenCastOp(ir::value* op, ir::type* type) {
  //TODO
  assert(false);
  return nullptr;
}

// Triton-IR Types
ir::type* Generator::GenIRType(::Type* type, ir::context& ctx) {
  if(auto T = type->ToVoid())
    return ir::type::get_void_ty(ctx);
  if(auto T = type->ToArithm())
    return GenIRArithmType(T, ctx);
  if(auto T = type->ToArray())
    return GenIRArrayType(T, ctx);
  if(auto T = type->ToTile())
    return GenIRTileType(T, ctx);
  if(auto T = type->ToFunc())
    return GenIRFuncType(T, ctx);
  if(auto T = type->ToPointer())
    return GenIRPointerType(T, ctx);
  if(auto T = type->ToStruct())
    return GenIRStructType(T, ctx);
  assert(false);
  return nullptr;
}

ir::type* Generator::GenIRArithmType(ArithmType* type, ir::context& ctx) {
  int tag = type->Tag();
  if(tag & T_BOOL)
    return ir::type::get_int1_ty(ctx);
  if(tag & T_CHAR)
    return ir::type::get_int8_ty(ctx);
  if(tag & T_SHORT)
    return ir::type::get_int16_ty(ctx);
  if(tag & T_INT)
    return ir::type::get_int32_ty(ctx);
  if(tag & T_LONG)
    return ir::type::get_int64_ty(ctx);
  if(tag & T_HALF)
    return ir::type::get_half_ty(ctx);
  if(tag & T_FLOAT)
    return ir::type::get_float_ty(ctx);
  if(tag & T_DOUBLE)
    return ir::type::get_double_ty(ctx);
  assert(false);
  return nullptr;
}

ir::type* Generator::GenIRArrayType(ArrayType* type, ir::context& ctx) {
  assert(false);
  return nullptr;
}

ir::type* Generator::GenIRTileType(TileType* type, ir::context& ctx) {
  ir::type* ele_ty = GenIRType(type->Derived().GetPtr(), ctx);
  auto _shape = type->Shape();
  ir::tile_type::tile_shapes_t shape;
  ir::type* int32_ty = ir::type::get_int32_ty(ctx);
  for(int s: _shape)
    shape.push_back(ir::constant_int::get(int32_ty, s));
  return ir::tile_type::get(ele_ty, shape);
}

ir::type* Generator::GenIRFuncType(FuncType* type, ir::context& ctx) {
  ir::type* ret_ty = GenIRType(type->Derived().GetPtr(), ctx);
  std::vector<ir::type*> param_tys;
  for(Object* obj: type->Params())
    param_tys.push_back(GenIRType(obj->Type(), ctx));
  return ir::function_type::get(ret_ty, param_tys);
}

ir::type* Generator::GenIRPointerType(PointerType* type, ir::context& ctx) {
  ir::type* ele_ty = GenIRType(type->Derived().GetPtr(), ctx);
  unsigned addr_space = 0;
  return ir::pointer_type::get(ele_ty, addr_space);
}

ir::type* Generator::GenIRStructType(StructType* type, ir::context& ctx) {
  assert(false);
  return nullptr;
}

void Generator::AllocObjects(Scope* scope, const FuncDef::ParamList& params) {
  return error_not_implemented();
}

// SSA
void Generator::pushScope() {
  mod_->add_new_scope();
}

void Generator::popScope() {
  mod_->pop_scope();
}

// LValue Generator
void LValAssigner::VisitBinaryOp(BinaryOp* binary) {
  error_not_implemented();
}

void LValAssigner::VisitUnaryOp(UnaryOp* unary) {
  if(unary->op_ != Token::DEREF)
    should_not_happen();
  gen_->VisitExpr(unary->operand_);
  ir::value* addr = gen_->ret_;
  ret_ = gen_->bld_->create_store(addr, rhs_);
}

void LValAssigner::VisitObject(Object* obj) {
  error_not_implemented();
}

void LValAssigner::VisitIdentifier(Identifier* ident) {
  std::string name = ident->Name();
  gen_->mod_->set_value(name, rhs_);
}



