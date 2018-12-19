#ifndef TDL_INCLUDE_AST_H
#define TDL_INCLUDE_AST_H

#include "parser.hpp"
#include <cassert>
#include <list>
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
  virtual void codegen(module*) { }
};

template<class T>
class list: public node {
public:
  list(const T& x): values_{x} {}

  node* append(const T& x){
    values_.push_back(x);
    return this;
  }

  void codegen(module* mod)
  { for(T x: values_){ x->codegen(mod); } }

  const std::list<T> &values() const
  { return values_; }

private:
  std::list<T> values_;
};

class binary_operator: public node{
public:
  binary_operator(BIN_OP_T op, node *lhs, node *rhs)
    : op_(op), lhs_(lhs), rhs_(rhs) { }

private:
  const BIN_OP_T op_;
  const node *lhs_;
  const node *rhs_;
};


class constant: public node{
public:
  constant(int value): value_(value) { }

private:
  const int value_;
};


class string_literal: public node{
public:
  string_literal(char *&value): value_(value) { }

public:
  std::string value_;
};

class unary_operator: public node{
public:
  unary_operator(UNARY_OP_T op, node *arg)
    : op_(op), arg_(arg) { }

private:
  const UNARY_OP_T op_;
  const node *arg_;
};

class cast_operator: public node{
public:
  cast_operator(node *type, node *arg): type_(type), arg_(arg) { }

public:
  const node *type_;
  const node *arg_;
};

class conditional_expression: public node{
public:
  conditional_expression(node *cond, node *true_value, node *false_value)
    : cond_(cond), true_value_(true_value), false_value_(false_value) { }

public:
  const node *cond_;
  const node *true_value_;
  const node *false_value_;
};

class assignment_expression: public node{
public:
  assignment_expression(node *lvalue, ASSIGN_OP_T op, node *rvalue)
    : lvalue_(lvalue), op_(op), rvalue_(rvalue) { }

public:
  ASSIGN_OP_T op_;
  const node *lvalue_;
  const node *rvalue_;
};

class statement: public node{

};

class declaration: public node{
public:
  declaration(node *spec, node *init)
    : spec_(spec), init_(init) { }

public:
  const node *spec_;
  const node *init_;
};


class compound_statement: public statement{
  typedef list<declaration*>* declarations_t;
  typedef list<statement*>* statements_t;

public:
  compound_statement(node* decls, node* statements)
    : decls_((declarations_t)decls), statements_((statements_t)statements) {}

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
  identifier(char *&name): declarator(nullptr), name_(name) { }
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

public:
  const list<parameter*>* args_;
};


class initializer : public declarator{
private:
  llvm::Type* type_impl(module* mod, llvm::Type *type) const;

public:
  initializer(node *id, node *initializer)
  : declarator(id), initializer_(initializer){ }


public:
  const node *initializer_;
};


class type: public node{
public:
  type(TYPE_T spec, node * decl)
    : spec_(spec), decl_((declarator*)decl) { }

public:
  const TYPE_T spec_;
  const declarator *decl_;
};

/* Function definition */
class function_definition: public node{
public:
  function_definition(node *spec, node *header, node *body)
    : spec_((declaration_specifier*)spec), header_((function *)header), body_((compound_statement*)body) { }

  void codegen(module* mod);

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

  void codegen(module* mod);

private:
  list<node*>* decls_;
};

}

}

#endif
