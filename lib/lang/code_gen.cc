#include "triton/lang/code_gen.h"
#include "triton/lang/evaluator.h"
#include "triton/lang/parser.h"
#include "triton/lang/token.h"
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
  auto type = binary->lhs_->Type()->ScalarType();
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
    case '@' : {
      ir::type* ret_ty = GenIRType(binary->Type(), *ctx_);
      ir::type* ret_scal_ty = ret_ty->get_scalar_ty();
      ir::value* _0;
      if(ret_scal_ty->is_float_ty())
        _0 = ir::constant_fp::get(ret_scal_ty, 0);
      else
        _0 = ir::constant_int::get(ret_scal_ty, 0);
      _0 = bld_->create_splat(_0, ret_ty->get_tile_shapes());
      return set_ret(bld_->create_dot(lhs, rhs, _0));
    }
    case Token::MASKED_DEREF: {
      ir::type* ret_ty = GenIRType(binary->Type(), *ctx_);
      return set_ret(bld_->create_masked_load(rhs, lhs, ir::undef_value::get(ret_ty)));
    }
    case Token::ELLIPSIS: {
      auto clhs = dynamic_cast<ir::constant_int*>(lhs);
      auto crhs = dynamic_cast<ir::constant_int*>(rhs);
      if(!clhs || !crhs)
        should_not_happen();
      return set_ret(bld_->insert(ir::make_range::create(clhs, crhs)));
    }
    case '+':
      if(binary->lhs_->Type()->ScalarType()->ToPointer()){
        return set_ret(bld_->create_gep(lhs, {rhs}));
      }
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

ir::reduce_inst::op_t reduce_op(int tag, bool is_float) {
  using ir::reduce_inst;
  switch(tag){
    case Token::ADD: return is_float ? reduce_inst::FADD : reduce_inst::ADD;
    case Token::SUB: return is_float ? reduce_inst::FSUB : reduce_inst::SUB;
    case Token::MAX: return is_float ? reduce_inst::FMAX : reduce_inst::MAX;
    case Token::MIN: return is_float ? reduce_inst::FMIN : reduce_inst::MIN;
    default: break;
  }
  should_not_happen();
  return reduce_inst::op_t();
}
void Generator::VisitUnaryOp(UnaryOp* unary) {
  // recursion
  Visit(unary->operand_);
  ir::value* arg = ret_;
  ir::type *arg_ty = arg->get_type();
  ir::type *arg_scal_ty = arg_ty->get_scalar_ty();
  // return
  switch  (unary->op_) {
    case Token::PREFIX_INC: return error_not_implemented();
    case Token::PREFIX_DEC: return error_not_implemented();
    case Token::POSTFIX_INC: return error_not_implemented();
    case Token::POSTFIX_DEC: return error_not_implemented();
    case Token::ADDR: return error_not_implemented();
    case Token::DEREF: return set_ret(bld_->create_load(arg));
    case Token::PLUS: return error_not_implemented();
    case Token::MINUS: return error_not_implemented();
    case '~': return set_ret(bld_->create_neg(arg));
    case '!': return set_ret(bld_->create_not(arg));
    case Token::CAST: return set_ret(GenCastOp(arg, GenIRType(unary->Type(), *ctx_)));
    case Token::REDUCE: {
      int ax, tag;
      UnaryOp::decodeRed(unary->info_, ax, tag);
      bool is_float = arg_scal_ty->is_floating_point_ty();
      ir::reduce_inst::op_t op = reduce_op(tag, is_float);
      return set_ret(bld_->create_reduce(arg, op, ax));
    }
    default: error_not_implemented();
  }
  return error_not_implemented();
}

void Generator::VisitTransOp(TransOp *trans) {
  Visit(trans->operand_);
  ir::value* arg = ret_;
  return set_ret(bld_->create_trans(arg, trans->getPerm()));
}

void Generator::VisitConditionalOp(ConditionalOp* condOp) {
//  auto &instructions = bld_->get_insert_block()->get_inst_list();
  VisitExpr(condOp->cond_);
  ir::value* cond = ret_;
  VisitExpr(condOp->exprTrue_);
  ir::value* true_val = ret_;
  VisitExpr(condOp->exprFalse_);
  ir::value* false_val = ret_;
  if(ir::load_inst* ld = dynamic_cast<ir::load_inst*>(true_val)) {
    ir::value* new_ld = bld_->create_masked_load(ld->get_pointer_operand(),
                                                  cond,
                                                  false_val);
    ld->replace_all_uses_with(new_ld);
    ld->erase_from_parent();
    return set_ret(new_ld);
  }
  return error_not_implemented();
}

void Generator::VisitFuncCall(FuncCall* funcCall) {
  std::string name = funcCall->Name();
  if(name == "get_program_id"){
    VisitExpr(funcCall->Args()->at(0));
    ir::value* ret = ret_;
    if(auto axis = dynamic_cast<ir::constant_int*>(ret))
      return set_ret(bld_->create_get_program_id(axis->get_value()));
    else
      return should_not_happen();
  }
  if(name == "sqrtf"){
    VisitExpr(funcCall->Args()->at(0));
    ir::value* ret = ret_;
    return set_ret(bld_->create_sqrt(ret));
  }
  return error_not_implemented();
}

void Generator::VisitObject(Object* obj) {
  return set_ret(mod_->get_value(obj->Name()));
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
    should_not_happen();
  if(inits.size() > 1)
    should_not_happen();
  if(inits.size() > 0)
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
  if (compoundStmt->scope_)
    pushScope();
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
    if(obj->Type()->ToPointer())
      fn->add_attr(i + 1, ir::attribute(ir::aligned, 16));
    for(ASTNode::Attr attr: obj->GetAttrList()){
      fn->add_attr(i + 1, GenIRAttr(attr));
    }
    if(obj->IsRestrictQualified())
      fn->add_attr(i, ir::attribute(ir::noalias));
    mod_->set_value(name, nullptr, args[i]);
    mod_->get_scope().types[name] = args[i]->get_type();
    i++;
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



ir::value* Generator::GenBroadcastOp(ir::value* src, ir::type* dst_ty) {
  if(src->get_type() == dst_ty)
    return src;
  if(dst_ty->is_tile_ty()) {
    ir::type *src_ty = src->get_type();
    auto dst_shapes = dst_ty->get_tile_shapes();
    if(!src_ty->is_tile_ty())
      return bld_->create_splat(src, dst_shapes);
    auto src_shapes = src_ty->get_tile_shapes();
    if(src_shapes.size() != dst_shapes.size()){
      unsigned src_numel = 1;
      for(unsigned s: src_shapes)
        src_numel *= s;
      unsigned dst_numel = 1;
      for(unsigned s: dst_shapes)
        dst_numel *= s;
      if(src_numel == dst_numel)
        return bld_->create_reshape(src, dst_shapes);
      else {
        auto padded_shapes = src_shapes;
        while(padded_shapes.size() != dst_shapes.size())
          padded_shapes.insert(padded_shapes.begin(), 1);
        // check that broadcast is legal
        for(size_t d = 0; d < padded_shapes.size(); d++){
          if(dst_shapes[d] != padded_shapes[d] &&
             padded_shapes[d] != 1)
            should_not_happen();
        }
        // pad and broadcast
        ir::value *padded = bld_->create_reshape(src, padded_shapes);
        return bld_->create_broadcast(padded, dst_shapes);
      }
    }
    else{
      return bld_->create_broadcast(src, dst_shapes);
    }
  }
  else if(src->get_type()->is_tile_ty() && src->get_type()->get_tile_num_elements() == 1){
    return bld_->create_downcast(src);
  }
  return src;
}

ir::value* Generator::GenNumcastOp(ir::value*src, ir::type* dst_ty) {
  ir::type *src_scalar_ty = src->get_type()->get_scalar_ty();
  ir::type *dst_scalar_ty = dst_ty->get_scalar_ty();
  if(src->get_type()->is_tile_ty())
    dst_ty = ir::tile_type::get_same_shapes(dst_scalar_ty, src->get_type());
  bool src_signed = false;
  bool dst_signed = false;
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

ir::value* Generator::GenCastOp(ir::value* src, ir::type* dst_ty) {
  return GenNumcastOp(GenBroadcastOp(src, dst_ty), dst_ty);
}

// Triton-IR Attr
ir::attribute Generator::GenIRAttr(ASTNode::Attr attr) {
  if(attr.kind == ASTNode::Attr::MULTIPLEOF) {
    VisitExpr(attr.vals[0]);
    auto cst = dynamic_cast<ir::constant_int*>(ret_);
    if(!cst) should_not_happen();
    return ir::attribute(ir::multiple_of, cst->get_value());
  }
  if(attr.kind == ASTNode::Attr::ALIGNED) {
    VisitExpr(attr.vals[0]);
    auto cst = dynamic_cast<ir::constant_int*>(ret_);
    return ir::attribute(ir::aligned, cst->get_value());
  }
  if(attr.kind == ASTNode::Attr::NOALIAS)
    return ir::attribute(ir::noalias);
  if(attr.kind == ASTNode::Attr::READONLY)
    return ir::attribute(ir::readonly);
  if(attr.kind == ASTNode::Attr::WRITEONLY)
    return ir::attribute(ir::writeonly);
  should_not_happen();
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
  for(int s: _shape)
    shape.push_back(static_cast<unsigned>(s));
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
  unsigned addr_space = 1;
  return ir::pointer_type::get(ele_ty, addr_space);
}

ir::type* Generator::GenIRStructType(StructType* type, ir::context& ctx) {
  error_not_implemented();
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
  if(binary->op_ != Token::MASKED_DEREF)
    error_not_implemented();
  gen_->VisitExpr(binary->lhs_);
  ir::value* mask = gen_->ret_;
  gen_->VisitExpr(binary->rhs_);
  ir::value* addr = gen_->ret_;
  ret_ = gen_->bld_->create_masked_store(addr, rhs_, mask);
}

void LValAssigner::VisitUnaryOp(UnaryOp* unary) {
  if(unary->op_ != Token::DEREF)
    should_not_happen();
  gen_->VisitExpr(unary->operand_);
  ir::value* addr = gen_->ret_;
  ret_ = gen_->bld_->create_store(addr, rhs_);
}

void LValAssigner::VisitObject(Object* obj) {
  std::string name = obj->Name();
  gen_->mod_->set_value(name, rhs_);
  ret_ = rhs_;
}

void LValAssigner::VisitIdentifier(Identifier* ident) {
  std::string name = ident->Name();
  gen_->mod_->set_value(name, rhs_);
  ret_ = rhs_;
}



