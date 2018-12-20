#ifndef TDL_INCLUDE_AST_H
#define TDL_INCLUDE_AST_H

#include "parser.hpp"
#include "llvm/IR/IRBuilder.h"
#include <cassert>
#include <vector>
#include <string>

namespace llvm{

class Function;
class Value;
class Type;

}

namespace tdl{

class module;

namespace ast{

// Enumerations
enum ASSIGN_OP_T{
  ASSIGN,
  INPLACE_MUL, INPLACE_DIV, INPLACE_MOD,
  INPLACE_ADD, INPLACE_SUB,
  INPLACE_LSHIFT, INPLACE_RSHIFT,
  INPLACE_AND, INPLACE_XOR,
  INPLACE_OR
};

enum BIN_OP_T{
  MUL, DIV, MOD,
  ADD, SUB,
  LEFT_SHIFT, RIGHT_SHIFT,
  LT, GT,
  LE, GE,
  EQ, NE,
  AND, XOR, OR,
  LAND, LOR
};

enum UNARY_OP_T{
  INC, DEC,
  PLUS, MINUS,
  ADDR, DEREF,
  COMPL, NOT
};

enum TYPE_T{
  VOID_T,
  UINT8_T, UINT16_T, UINT32_T, UINT64_T,
  INT8_T, INT16_T, INT32_T, INT64_T,
  FLOAT32_T, FLOAT64_T
};

// AST
class node {
public:
  virtual llvm::Value* codegen(module*) const { return nullptr; }
};

template<class T>
class list: public node {
public:
  list(const T& x): values_{x} {}

  node* append(const T& x){
    values_.push_back(x);
    return this;
  }

  llvm::Value* codegen(module* mod) const{
    for(T x: values_){
      x->codegen(mod);
    }
    return nullptr;
  }

  const std::vector<T> &values() const
  { return values_; }

private:
  std::vector<T> values_;
};

class expression: public node{
public:
  virtual llvm::Value* codegen(module *) const = 0;
};

class binary_operator: public expression{
private:
  llvm::Value* llvm_op(llvm::IRBuilder<> &bld, llvm::Value *lhs, llvm::Value *rhs, const std::string &name) const;

public:
  binary_operator(BIN_OP_T op, node *lhs, node *rhs)
    : op_(op), lhs_((expression*)lhs), rhs_((expression*)rhs) { }
  llvm::Value* codegen(module *) const;

private:
  const BIN_OP_T op_;
  const expression *lhs_;
  const expression *rhs_;
};


class constant: public expression{
public:
  constant(int value): value_(value) { }
  llvm::Value* codegen(module *mod) const;

private:
  const int value_;
};


class string_literal: public expression{
public:
  string_literal(char *&value): value_(value) { }
  llvm::Value* codegen(module *mod) const;

public:
  std::string value_;
};

class unary_operator: public expression{
private:
  llvm::Value *llvm_op(llvm::IRBuilder<> &builder, llvm::Value *arg, const std::string &name) const;

public:
  unary_operator(UNARY_OP_T op, node *arg)
    : op_(op),
      arg_((expression*)arg) { }

  llvm::Value* codegen(module *mod) const;

private:
  const UNARY_OP_T op_;
  const expression *arg_;
};

class type_name;
class cast_operator: public expression{
private:
  llvm::Value *llvm_op(llvm::IRBuilder<> &builder, llvm::Type *T, llvm::Value *arg, const std::string &name) const;

public:
  cast_operator(node *T, node *arg):
    T_((type_name*)T),
    arg_((expression*)arg) { }

  llvm::Value* codegen(module *mod) const;

public:
  const type_name *T_;
  const expression *arg_;
};

class conditional_expression: public expression{
private:
  llvm::Value *llvm_op(llvm::IRBuilder<> &builder,
                       llvm::Value *cond, llvm::Value *true_value, llvm::Value *false_value,
                       const std::string &name) const;

public:
  conditional_expression(node *cond, node *true_value, node *false_value)
    : cond_((expression*)cond),
      true_value_((expression*)true_value),
      false_value_((expression*)false_value) { }

  llvm::Value* codegen(module *mod) const;

public:
  const expression *cond_;
  const expression *true_value_;
  const expression *false_value_;
};

class assignment_expression: public expression{
private:
  llvm::Value *llvm_op(llvm::IRBuilder<> &builder,
                       llvm::Value *lvalue, llvm::Value *rvalue,
                       const std::string &name) const;

public:
  assignment_expression(node *lvalue, ASSIGN_OP_T op, node *rvalue)
    : lvalue_((expression*)lvalue), op_(op), rvalue_((expression*)rvalue) { }

  llvm::Value* codegen(module *mod) const;

public:
  ASSIGN_OP_T op_;
  const expression *lvalue_;
  const expression *rvalue_;
};

class statement: public node{
};

class initializer;
class declaration_specifier;

class declaration: public node{
public:
  declaration(node *spec, node *init)
    : spec_((declaration_specifier*)spec), init_((list<initializer*>*)init) { }

  llvm::Value* codegen(module* mod) const;

public:
  const declaration_specifier *spec_;
  const list<initializer*> *init_;
};


class compound_statement: public statement{
  typedef list<declaration*>* declarations_t;
  typedef list<statement*>* statements_t;

public:
  compound_statement(node* decls, node* statements)
    : decls_((declarations_t)decls), statements_((statements_t)statements) {}

  llvm::Value* codegen(module* mod) const;

private:
  declarations_t decls_;
  statements_t statements_;
};

class selection_statement: public statement{
public:
  selection_statement(node *cond, node *if_value, node *else_value = nullptr)
    : cond_(cond), if_value_(if_value), else_value_(else_value) { }

public:
  const node *cond_;
  const node *if_value_;
  const node *else_value_;
};

class iteration_statement: public statement{
public:
  iteration_statement(node *init, node *stop, node *exec, node *statements)
    : init_(init), stop_(stop), exec_(exec), statements_(statements) { }

private:
  const node *init_;
  const node *stop_;
  const node *exec_;
  const node *statements_;
};

class no_op: public statement { };

// Types

class declaration_specifier: public node{
public:
  declaration_specifier(TYPE_T spec)
    : spec_(spec) { }

  llvm::Type* type(module *mod) const;

private:
  const TYPE_T spec_;
};

class declarator;
class parameter: public node {
public:
  parameter(node *spec, node *decl)
    : spec_((declaration_specifier*)spec),
      decl_((declarator*)decl) { }

  llvm::Type* type(module *mod) const;
  std::string name() const;

public:
  const declaration_specifier *spec_;
  const declarator *decl_;
};

/* Declarators */
class pointer;
class identifier;

class declarator: public node{
  virtual llvm::Type* type_impl(module*mod, llvm::Type *type) const = 0;

public:
  declarator(node *lhs)
    : lhs_((declarator*)lhs), ptr_(nullptr){ }

  llvm::Type* type(module*mod, llvm::Type *type) const;

  const identifier* id() const {
    return (const identifier*)lhs_;
  }

  declarator *set_ptr(node *ptr){
    ptr_ = (pointer*)ptr;
    return this;
  }

protected:
  declarator *lhs_;
  pointer *ptr_;
};

class identifier: public declarator{
  llvm::Type* type_impl(module*mod, llvm::Type *type) const;

public:
  identifier(char *&name): declarator(this), name_(name) { }
  const std::string &name() const;

private:
  std::string name_;
};

class pointer: public declarator{
private:
  llvm::Type* type_impl(module *mod, llvm::Type *type) const;

public:
  pointer(node *id): declarator(id) { }
};

class tile: public declarator{
private:
  llvm::Type* type_impl(module *mod, llvm::Type *type) const;

public:
  tile(node *id, node *shapes)
    : declarator(id), shapes_((list<constant*>*)(shapes)) { }

public:
  const list<constant*>* shapes_;
};

class function: public declarator{
private:
  llvm::Type* type_impl(module *mod, llvm::Type *type) const;

public:
  function(node *id, node *args)
    : declarator(id), args_((list<parameter*>*)args) { }

  void bind_parameters(module *mod, llvm::Function *fn) const;

public:
  const list<parameter*>* args_;
};


class initializer : public declarator{
private:
  llvm::Type* type_impl(module* mod, llvm::Type *type) const;

public:
  initializer(node *decl, node *init)
  : declarator((node*)((declarator*)decl)->id()),
    decl_((declarator*)decl), init_((expression*)init){ }

  void specifier(const declaration_specifier *spec);
  llvm::Value* codegen(module *) const;

public:
  const declaration_specifier *spec_;
  const declarator *decl_;
  const expression *init_;
};


class type_name: public node{
public:
  type_name(node *spec, node * decl)
    : spec_((declaration_specifier*)spec), decl_((declarator*)decl) { }

  llvm::Type *type(module *mod) const;

public:
  const declaration_specifier *spec_;
  const declarator *decl_;
};

/* Function definition */
class function_definition: public node{
public:
  function_definition(node *spec, node *header, node *body)
    : spec_((declaration_specifier*)spec), header_((function *)header), body_((compound_statement*)body) { }

  llvm::Value* codegen(module* mod) const;

public:
  const declaration_specifier *spec_;
  const function *header_;
  const compound_statement *body_;
};

/* Translation Unit */
class translation_unit: public node{
public:
  translation_unit(node *item)
    : decls_((list<node*>*)item) { }

  translation_unit *add(node *item) {
    decls_->append(item);
    return this;
  }

  llvm::Value* codegen(module* mod) const;

private:
  list<node*>* decls_;
};

}

}

#endif
