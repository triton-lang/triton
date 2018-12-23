#include <functional>
#include "ast.h"
#include "codegen.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/CFG.h"

using namespace llvm;

namespace tdl{

/* Context */
context::context() { }

LLVMContext *context::handle() {
  return &handle_;
}

/* Module */
module::module(const std::string &name, context *ctx)
  : handle_(name.c_str(), *ctx->handle()), builder_(*ctx->handle()) {
}

llvm::Module* module::handle() {
  return &handle_;
}

llvm::IRBuilder<>& module::builder() {
  return builder_;
}

void module::set_value(const ast::node *node, BasicBlock *block, Value *value){
  values_[val_key_t{node, block}] = value;
}

void module::set_value(const ast::node* node, llvm::Value* value){
  return set_value(node, builder_.GetInsertBlock(), value);
}

PHINode* module::make_phi(Type *type, unsigned num_values, BasicBlock *block){
  llvm::BasicBlock::iterator save = builder_.GetInsertPoint();
  builder_.SetInsertPoint(&*block->getFirstInsertionPt());
  PHINode *res = builder_.CreatePHI(type, num_values);
  builder_.SetInsertPoint(&*save);
  return res;
}

Value *module::add_phi_operands(const ast::node *node, PHINode *&phi){
  BasicBlock *block = phi->getParent();
  for(BasicBlock *pred: predecessors(block)){
    llvm::Value *value = get_value(node, pred);
    if(phi->getType()==nullptr){
      phi = make_phi(value->getType(), pred_size(block), block);
    }
    phi->addIncoming(value, pred);
  }
}

llvm::Value *module::get_value_recursive(const ast::node* node, BasicBlock *block) {
  llvm::Value *result;
  if(sealed_blocks_.find(block) == sealed_blocks_.end()){
    incomplete_phis_[block][node] = make_phi(nullptr, 1, block);
  }
  else if(pred_size(block) <= 1){
    result = get_value(node, *pred_begin(block));
  }
  else{
    result = make_phi(nullptr, 1, block);
    set_value(node, block, result);
    add_phi_operands(node, (PHINode*&)result);
  }
  set_value(node, block, result);
}

llvm::Value *module::get_value(const ast::node* node, BasicBlock *block) {
  val_key_t key(node, block);
  if(values_.find(key) != values_.end())
    return values_.at(key);
  return get_value_recursive(node, block);
}

llvm::Value *module::get_value(const ast::node *node) {
  return get_value(node, builder_.GetInsertBlock());
}

llvm::Value *module::seal_block(BasicBlock *block){
  for(auto &x: incomplete_phis_[block])
    add_phi_operands(x.first, x.second);
  sealed_blocks_.insert(block);
}

namespace ast{

/* Translation unit */
Value* translation_unit::codegen(module *mod) const{
  decls_->codegen(mod);
  return nullptr;
}

/* Declaration specifier */
Type* declaration_specifier::type(module *mod) const {
  LLVMContext &ctx = mod->handle()->getContext();
  switch (spec_) {
  case VOID_T:      return Type::getVoidTy(ctx);
  case INT8_T:      return IntegerType::get(ctx, 8);
  case INT16_T:     return IntegerType::get(ctx, 16);
  case INT32_T:     return IntegerType::get(ctx, 32);
  case INT64_T:     return IntegerType::get(ctx, 64);
  case FLOAT32_T:   return Type::getFloatTy(ctx);
  case FLOAT64_T:   return Type::getDoubleTy(ctx);
  default: assert(false && "unreachable"); throw;
  }
}

/* Parameter */
Type* parameter::type(module *mod) const {
  return decl_->type(mod, spec_->type(mod));
}

const identifier *parameter::id() const {
  return decl_->id();
}

/* Declarators */
Type* declarator::type(module *mod, Type *type) const{
  if(ptr_)
    return type_impl(mod, ptr_->type(mod, type));
  return type_impl(mod, type);
}

// Identifier
Type* identifier::type_impl(module *, Type *type) const{
  return type;
}

const std::string &identifier::name() const{
  return name_;
}

// Tile
Type* tile::type_impl(module*, Type *type) const{
  return TileType::get(type, shapes_->values().size());
}

// Initializer
Type* initializer::type_impl(module *mod, Type *type) const{
  return decl_->type(mod, type);
}

// Pointer
Type* pointer::type_impl(module*, Type *type) const{
  return PointerType::get(type, 1);
}

// Function
void function::bind_parameters(module *mod, Function *fn) const{
  std::vector<llvm::Value*> args;
  std::transform(fn->arg_begin(), fn->arg_end(), std::back_inserter(args), [&](llvm::Argument& x){ return &x;});
  assert(args.size() == args_->values().size());
  for(size_t i = 0; i < args.size(); i++){
    parameter *param_i = args_->values().at(i);
    const identifier *id_i = param_i->id();
    if(id_i){
      args[i]->setName(id_i->name());
      mod->set_value(id_i, nullptr, args[i]);
    }
  }
}

Type* function::type_impl(module*mod, Type *type) const{
  SmallVector<Type*, 8> types;
  for(parameter* param: args_->values()){
    types.push_back(param->type(mod));
  }
  return FunctionType::get(type, types, false);
}

/* Function definition */
Value* function_definition::codegen(module *mod) const{
  FunctionType *prototype = (FunctionType *)header_->type(mod, spec_->type(mod));
  const std::string &name = header_->id()->name();
  Function *fn = Function::Create(prototype, Function::ExternalLinkage, name, mod->handle());
  header_->bind_parameters(mod, fn);
  BasicBlock *entry = BasicBlock::Create(mod->handle()->getContext(), "entry", fn);
  mod->builder().SetInsertPoint(entry);
  body_->codegen(mod);
  return nullptr;
}

/* Statements */
Value* compound_statement::codegen(module* mod) const{
  decls_->codegen(mod);
  if(statements_)
    statements_->codegen(mod);
  return nullptr;
}

/* Declaration */
Value* declaration::codegen(module* mod) const{
  for(initializer *init: init_->values())
    init->specifier(spec_);
  init_->codegen(mod);
  return nullptr;
}

/* Initializer */
void initializer::specifier(const declaration_specifier *spec) {
  spec_ = spec;
}

Value* initializer::codegen(module * mod) const{
  Type *ty = decl_->type(mod, spec_->type(mod));
  std::string name = decl_->id()->name();
  Value *value = llvm::UndefValue::get(ty);
  value->setName(name);
  return nullptr;
}

/*------------------*/
/*    Expression    */
/*------------------*/
llvm::Value *llvm_cast(llvm::IRBuilder<> &builder, Value *src, Type *dst_ty){
  Type *src_ty = src->getType();
  bool src_signed = false;
  bool dst_signed = false;
  if(src_ty == dst_ty)
    return src;
  else if(src_ty->isIntegerTy() && src_signed && dst_ty->isFloatingPointTy())
    return builder.CreateSIToFP(src, dst_ty);

  else if(src_ty->isIntegerTy() && !src_signed && dst_ty->isFloatingPointTy())
    return builder.CreateUIToFP(src, dst_ty);

  else if(src_ty->isFloatingPointTy() && dst_ty->isIntegerTy() && dst_signed)
    return builder.CreateFPToSI(src, dst_ty);

  else if(src_ty->isFloatingPointTy() && dst_ty->isIntegerTy() && !dst_signed)
    return builder.CreateFPToUI(src, dst_ty);

  else if(src_ty->isFloatingPointTy() && dst_ty->isFloatingPointTy() &&
          src_ty->getFPMantissaWidth() < dst_ty->getFPMantissaWidth())
    return builder.CreateFPExt(src, dst_ty);

  else if(src_ty->isFloatingPointTy() && dst_ty->isFloatingPointTy() &&
          src_ty->getFPMantissaWidth() > dst_ty->getFPMantissaWidth())
    return builder.CreateFPTrunc(src, dst_ty);

  else if(src_ty->isIntegerTy() && dst_ty->isIntegerTy() &&
          src_ty->getIntegerBitWidth())
    return builder.CreateIntCast(src, dst_ty, dst_signed);

  else{
    assert(false && "unreachable");
    throw;
  }
}

inline void implicit_cast(llvm::IRBuilder<> &builder, Value *&lhs, Value *&rhs,
                          bool &is_float, bool &is_ptr, bool &is_int, bool &is_signed){
  // Input types
  Type *left_ty = lhs->getType();
  Type *right_ty = rhs->getType();
  // One operand is pointer
  if(left_ty->isPointerTy()){
    is_ptr = true;
  }
  // One operand is double
  else if(left_ty->isDoubleTy() || right_ty->isDoubleTy()){
    Value *&to_convert = left_ty->isDoubleTy()?rhs:lhs;
    to_convert = llvm_cast(builder, to_convert, builder.getDoubleTy());
    is_float = true;
  }
  // One operand is float
  else if(left_ty->isFloatTy() || right_ty->isFloatTy()){
    Value *&to_convert = left_ty->isFloatTy()?rhs:lhs;
    to_convert = llvm_cast(builder, to_convert, builder.getFloatTy());
    is_float = true;
  }
  // Both operands are integers
  else if(left_ty->isIntegerTy() && right_ty->isIntegerTy()){
    is_int = true;
    is_signed = false;
    if(left_ty->getIntegerBitWidth() != right_ty->getIntegerBitWidth()){
      Value *&to_convert = (left_ty->getIntegerBitWidth() > right_ty->getIntegerBitWidth())?rhs:lhs;
      Type *dst_ty = (to_convert==lhs)?right_ty:left_ty;
      to_convert = llvm_cast(builder, to_convert, dst_ty);
    }
  }
  // Not reachable
  else{
    assert(false);
    throw;
  }
}

//inline void implicit_broadcast(llvm::IRBuilder<> &builder, Value *&lhs, Value *&rhs){
//  return;
//}

/* Binary operator */
Value *binary_operator::llvm_op(llvm::IRBuilder<> &builder, Value *lhs, Value *rhs, const std::string &name) const
{
  bool is_float, is_ptr, is_int, is_signed;
  implicit_cast(builder, lhs, rhs, is_float, is_ptr, is_int, is_signed);
//  implicit_broadcast(builder, lhs, rhs);
  // Mul
  if(op_==MUL && is_float)
    return builder.CreateFMul(lhs, rhs, name);
  if(op_==MUL && is_int)
    return builder.CreateMul(lhs, rhs, name);
  // Div
  if(op_==DIV && is_float)
    return builder.CreateFDiv(lhs, rhs, name);
  if(op_==DIV && is_int && is_signed)
    return builder.CreateSDiv(lhs, rhs, name);
  if(op_==DIV && is_int && !is_signed)
    return builder.CreateUDiv(lhs, rhs, name);
  // Mod
  if(op_==MOD && is_float)
    return builder.CreateFRem(lhs, rhs, name);
  if(op_==MOD && is_int && is_signed)
    return builder.CreateSRem(lhs, rhs, name);
  if(op_==MOD && is_int && !is_signed)
    return builder.CreateURem(lhs, rhs, name);
  // Add
  if(op_==ADD && is_float)
    return builder.CreateFAdd(lhs, rhs, name);
  if(op_==ADD && is_int)
    return builder.CreateAdd(lhs, rhs);
  if(op_==ADD && is_ptr)
    return builder.CreateGEP(lhs, {rhs});
  // Sub
  if(op_==SUB && is_float)
    return builder.CreateFSub(lhs, rhs, name);
  if(op_==SUB && is_int)
    return builder.CreateSub(lhs, rhs, name);
  if(op_==SUB && is_ptr)
    return builder.CreateGEP(lhs, {builder.CreateNeg(rhs)});
  // Left shift
  if(op_==LEFT_SHIFT){
    assert(is_int);
    return builder.CreateLShr(lhs, rhs, name);
  }
  // Right shift
  if(op_==RIGHT_SHIFT){
    assert(is_int);
    return builder.CreateAShr(lhs, rhs, name);
  }
  // LT
  if(op_ == LT && is_float)
    return builder.CreateFCmpOLT(lhs, rhs, name);
  if(op_ == LT && is_int && is_signed)
    return builder.CreateICmpSLT(lhs, rhs, name);
  if(op_ == LT && is_int && !is_signed)
    return builder.CreateICmpULT(lhs, rhs, name);
  // GT
  if(op_ == GT && is_float)
    return builder.CreateFCmpOGT(lhs, rhs, name);
  if(op_ == GT && is_int && is_signed)
    return builder.CreateICmpSGT(lhs, rhs, name);
  if(op_ == GT && is_int && !is_signed)
    return builder.CreateICmpUGT(lhs, rhs, name);
  // LE
  if(op_ == LE && is_float)
    return builder.CreateFCmpOLE(lhs, rhs, name);
  if(op_ == LE && is_int && is_signed)
    return builder.CreateICmpSLE(lhs, rhs, name);
  if(op_ == LE && is_int && !is_signed)
    return builder.CreateICmpULE(lhs, rhs, name);
  // GE
  if(op_ == GE && is_float)
    return builder.CreateFCmpOGE(lhs, rhs, name);
  if(op_ == GE && is_int && is_signed)
    return builder.CreateICmpSGE(lhs, rhs, name);
  if(op_ == GE && is_int && !is_signed)
    return builder.CreateICmpUGE(lhs, rhs, name);
  // EQ
  if(op_ == EQ && is_float)
    return builder.CreateFCmpOEQ(lhs, rhs, name);
  if(op_ == EQ && is_int)
    return builder.CreateICmpEQ(lhs, rhs, name);
  // NE
  if(op_ == NE && is_float)
    return builder.CreateFCmpONE(lhs, rhs, name);
  if(op_ == NE && is_int)
    return builder.CreateICmpNE(lhs, rhs, name);
  // AND
  if(op_ == AND){
    assert(is_int);
    return builder.CreateAnd(lhs, rhs, name);
  }
  if(op_ == XOR){
    assert(is_int);
    return builder.CreateXor(lhs, rhs, name);
  }
  if(op_ == OR){
    assert(is_int);
    return builder.CreateOr(lhs, rhs, name);
  }
  if(op_ == LAND){
    assert(is_int);
    return builder.CreateAnd(lhs, rhs, name);
  }
  if(op_ == LOR){
    assert(is_int);
    return builder.CreateOr(lhs, rhs, name);
  }
  assert(false && "unreachable");
  throw;
}

Value* binary_operator::codegen(module *mod) const{
  Value *lhs = lhs_->codegen(mod);
  Value *rhs = rhs_->codegen(mod);
  Value *result = llvm_op(mod->builder(), lhs, rhs, "");
  return result;
}

/* Unary operator */
Value *unary_operator::llvm_op(llvm::IRBuilder<> &builder, Value *arg, const std::string &name) const{
  Type *atype = arg->getType();
  bool is_float = atype->isFloatingPointTy();
  bool is_int = atype->isIntegerTy();
  if(op_ == INC){
    assert(is_int);
    return builder.CreateAdd(arg, builder.getInt32(1), name);
  }
  if(op_ == DEC){
    assert(is_int);
    return builder.CreateSub(arg, builder.getInt32(1), name);
  }
  if(op_ == PLUS)
    return arg;
  if(op_ == MINUS && is_float)
    return builder.CreateFNeg(arg, name);
  if(op_ == MINUS && is_int)
    return builder.CreateNeg(arg, name);
  if(op_ == ADDR)
    throw std::runtime_error("not supported");
  if(op_ == DEREF)
    return builder.CreateLoad(arg, name);
  if(op_ == COMPL)
    throw std::runtime_error("not supported");
  if(op_ == NOT)
    return builder.CreateNot(arg, name);
  assert(false && "unrechable");
  throw;
}

Value* unary_operator::codegen(module *mod) const{
  Value *arg = arg_->codegen(mod);
  Value *result = llvm_op(mod->builder(), arg, "");
  return result;
}

/* Cast operator */
Value *cast_operator::llvm_op(IRBuilder<> &builder, Type *T, Value *arg, const std::string &name) const{
  return nullptr;
}

Value* cast_operator::codegen(module *mod) const{
  Value *arg = arg_->codegen(mod);
  Type *T = T_->type(mod);
  return llvm_op(mod->builder(), T, arg, "");
}

/* Conditional expression */
Value *conditional_expression::llvm_op(IRBuilder<> &builder, Value *cond, Value *true_value, Value *false_value, const std::string &name) const{
  return nullptr;
}

Value *conditional_expression::codegen(module *mod) const{
  Value *cond = cond_->codegen(mod);
  Value *true_value = true_value_->codegen(mod);
  Value *false_value = false_value_->codegen(mod);
  return llvm_op(mod->builder(), cond, true_value, false_value, "");
}

/* Assignment expression */
Value *assignment_expression::codegen(module *mod) const{
  Value *rvalue = rvalue_->codegen(mod);
  mod->set_value(lvalue_, rvalue);
  return rvalue;
}

/* Type name */
llvm::Type *type_name::type(module *mod) const{
  return decl_->type(mod, spec_->type(mod));
}

/* String literal */
llvm::Value* string_literal::codegen(module *mod) const{
  return ConstantDataArray::getString(mod->handle()->getContext(), value_);
}

/* Constant */
llvm::Value* constant::codegen(module *mod) const{
  return mod->builder().getInt32(value_);
}

/* Named */
llvm::Value* named_expression::codegen(module *mod) const{
  return mod->get_value(id_);
}


}

}
