#include "triton/lang/wgtcc/code_gen.h"
#include "triton/lang/wgtcc/evaluator.h"
#include "triton/lang/wgtcc/parser.h"
#include "triton/lang/wgtcc/token.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"

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
    case Token::ELLIPSIS: {
      auto clhs = dynamic_cast<ir::constant_int*>(lhs);
      auto crhs = dynamic_cast<ir::constant_int*>(rhs);
      if(!clhs || !crhs)
        should_not_happen();
      return set_ret(ir::constant_range::get(clhs, crhs));
    }
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
  std::cout << obj->Name() << " " << val->get_type()->get_type_id() << " " << ty->get_type_id() << std::endl;
  if(val->get_type()->is_tile_ty() && ty->is_tile_ty()) {
    for(auto s: val->get_type()->get_tile_shapes())
      std::cout << s->get_value() << std::endl;
    std::cout << "---" << std::endl;
    for(auto s: ty->get_tile_shapes())
      std::cout << s->get_value() << std::endl;
  }
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

void Generator::VisitForStmt(ForStmt *forStmt) {
  Stmt *init_ = forStmt->init_;
  Expr *cond_ = forStmt->cond_;
  Expr *step_ = forStmt->step_;
  Stmt *body_ = forStmt->body_;
  ir::basic_block *current_bb = bld_->get_insert_block();
  ir::function *fn = current_bb->get_parent();
  ir::basic_block *loop_bb = ir::basic_block::create(*ctx_, "loop", fn);
  ir::basic_block *next_bb = ir::basic_block::create(*ctx_, "postloop", fn);
  mod_->set_continue_fn([&](){
    if(step_)
      VisitExpr(step_);
    VisitExpr(cond_);
    ir::value *cond = ret_;
    return bld_->create_cond_br(cond, loop_bb, next_bb);
  });
  VisitStmt(init_);
  VisitExpr(cond_);
  ir::value *cond = ret_;
  bld_->create_cond_br(cond, loop_bb, next_bb);
  bld_->set_insert_point(loop_bb);
  VisitStmt(body_);
  if(!is_terminator(ret_))
    mod_->get_continue_fn()();
  ir::basic_block *stop_bb = bld_->get_insert_block();
  mod_->seal_block(stop_bb);
  mod_->seal_block(loop_bb);
  mod_->seal_block(bld_->get_insert_block());
  mod_->seal_block(next_bb);
  bld_->set_insert_point(next_bb);
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
//    AllocObjects(compoundStmt->scope_);
    pushScope();
  }
  for (auto stmt: compoundStmt->stmts_)
    Visit(stmt);
  if(compoundStmt->scope_)
    popScope();
}

void Generator::VisitFuncDef(FuncDef* funcDef) {
  Stmt *body = funcDef->body_;
  const std::string& name = funcDef->Name();
  FuncType* type = funcDef->FuncType();
  auto prototype = dynamic_cast<ir::function_type*>(GenIRType(type, *ctx_));
  if(!prototype)
    should_not_happen();
  ir::function *fn = mod_->get_or_insert_function(name, prototype);
  std::vector<ir::argument*> args = fn->args();
  size_t i = 0;
  for(Object* obj: type->Params()){
    std::string name = obj->Name();
    args[i]->set_name(name);
    mod_->set_value(name, nullptr, args[i]);
    mod_->get_scope().types[name] = args[i]->get_type();
  }
  ir::basic_block *entry = ir::basic_block::create(mod_->get_context(), "entry", fn);
  mod_->seal_block(entry);
  mod_->get_builder().set_insert_point(entry);
  VisitStmt(body);
  if(!dynamic_cast<ir::return_inst*>(ret_))
    mod_->get_builder().create_ret_void();
}

void Generator::VisitTranslationUnit(TranslationUnit* unit) {
  pushScope();
  for (auto extDecl: unit->ExtDecls())
    Visit(extDecl);
  popScope();
}

void Generator::Gen(ir::module *mod) {
  mod_ = mod;
  ctx_ = &mod_->get_context();
  bld_ = &mod_->get_builder();
  assign_ = new LValAssigner(this);
  VisitTranslationUnit(parser_->Unit());
  delete assign_;
  assign_ = nullptr;
}


// Triton-IR Values

ir::value* Generator::GenCastOp(ir::value* src, ir::type* dst_ty) {
  if(dst_ty->is_tile_ty()) {
    auto dst_shapes = dst_ty->get_tile_shapes();
    if(!src->get_type()->is_tile_ty())
      return bld_->create_splat(src, dst_shapes);
    auto src_shapes = src->get_type()->get_tile_shapes();
    if(src_shapes.size() != dst_shapes.size())
      return bld_->create_reshape(src, dst_shapes);
    else
      return bld_->create_broadcast(src, dst_shapes);
  }
  ir::type *src_scalar_ty = src->get_type()->get_scalar_ty();
  ir::type *dst_scalar_ty = dst_ty->get_scalar_ty();
  bool src_signed = false;
  bool dst_signed = false;

  if(src->get_type()->is_tile_ty())
    dst_ty = ir::tile_type::get_same_shapes(dst_scalar_ty, src->get_type());

  if(src_scalar_ty == dst_scalar_ty)
    return src;

  else if(src_scalar_ty->is_integer_ty() && src_signed && dst_scalar_ty->is_floating_point_ty())
    return bld_->create_si_to_fp(src, dst_ty);

  else if(src_scalar_ty->is_integer_ty() && !src_signed && dst_scalar_ty->is_floating_point_ty())
    return bld_->create_ui_to_fp(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty() && dst_signed)
    return bld_->create_fp_to_si(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_integer_ty() && !dst_signed)
    return bld_->create_fp_to_ui(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_floating_point_ty() &&
          src_scalar_ty->get_fp_mantissa_width() < dst_scalar_ty->get_fp_mantissa_width())
    return bld_->create_fp_ext(src, dst_ty);

  else if(src_scalar_ty->is_floating_point_ty() && dst_scalar_ty->is_floating_point_ty() &&
          src_scalar_ty->get_fp_mantissa_width() > dst_scalar_ty->get_fp_mantissa_width())
    return bld_->create_fp_trunc(src, dst_ty);

  else if(src_scalar_ty->is_integer_ty() && dst_scalar_ty->is_integer_ty() &&
          src_scalar_ty->get_integer_bitwidth())
    return bld_->create_int_cast(src, dst_ty, dst_signed);

  else{
    should_not_happen();
    return nullptr;
  }
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



