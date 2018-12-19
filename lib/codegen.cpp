#include "ast.h"
#include "codegen.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"

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


namespace ast{

/* Translation unit */
void translation_unit::codegen(module *mod){
  decls_->codegen(mod);
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
Type* initializer::type_impl(module *, Type *type) const{
  return type;
}

// Pointer
Type* pointer::type_impl(module*, Type *type) const{
  return PointerType::get(type, 1);
}

// Function
Type* function::type_impl(module*mod, Type *type) const{
  SmallVector<Type*, 8> types;
  for(parameter* param: args_->values()){
    types.push_back(param->type(mod));
  }
  return FunctionType::get(type, types, false);
}

/* Function definition */
void function_definition::codegen(module *mod){
  llvm::FunctionType *prototype = (llvm::FunctionType *)header_->type(mod, spec_->type(mod));
  const std::string &name = header_->id()->name();
  llvm::Function *fn = llvm::Function::Create(prototype, llvm::Function::ExternalLinkage, name, mod->handle());
  llvm::BasicBlock::Create(mod->handle()->getContext(), "entry", fn);
  mod->builder().SetInsertPoint();

}

}

}
