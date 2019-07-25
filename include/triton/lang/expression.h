#ifndef TDL_INCLUDE_LANG_EXPRESSION_H
#define TDL_INCLUDE_LANG_EXPRESSION_H

#include "lang.h"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>


namespace triton{


namespace ir{
  class function;
  class value;
  class type;
  class builder;
  class module;
}

namespace lang{


enum slice_enum_t{
  ALL,
  NEWAXIS
};

class slice: public node{
public:
  slice(slice_enum_t type)
    : type_(type){}

  slice_enum_t type() const{
    return type_;
  }

public:
  const slice_enum_t type_;
};


class named_expression;

class expression: public node{
public:
  virtual ir::value* codegen(ir::module *) const = 0;
  named_expression *lvalue() const { return lvalue_; }

protected:
  named_expression *lvalue_;
};

class postfix_expression: public expression{

};

class builtin_expression: public node{

};

class typed_declaration_specifier;
class alloc_const_expression: public builtin_expression{
public:
  alloc_const_expression(node *spec, node *size): spec_((typed_declaration_specifier*)spec), size_((constant*)size) { }
  ir::value* codegen(ir::module *mod) const;

private:
  const typed_declaration_specifier* spec_;
  const constant* size_;
};

class get_range_id_expression: public builtin_expression{
public:
  get_range_id_expression(node *axis): axis_((constant*)axis) { }
  ir::value* codegen(ir::module *) const;

private:
  const constant* axis_;
};

class atomic_cas_expression: public builtin_expression{
public:
  atomic_cas_expression(node *ptr, node *cmp, node *val): ptr_(ptr), cmp_(cmp), val_(val) { }
  ir::value* codegen(ir::module *) const;

private:
  const node *ptr_;
  const node *cmp_;
  const node *val_;
};

class atomic_add_expression: public builtin_expression{
public:
  atomic_add_expression(node *ptr, node *val): ptr_(ptr), val_(val) { }
  ir::value* codegen(ir::module *) const;

private:
  const node *ptr_;
  const node *val_;
};


class matmul_expression: public builtin_expression{
public:
  matmul_expression(node* A, node *B, node *C):
    A_((expression*)A), B_((expression*)B), C_((expression*)C) { }
  ir::value* codegen(ir::module *) const;

private:
  const expression *A_;
  const expression *B_;
  const expression *C_;
};

class max_expression: public builtin_expression{
public:
  max_expression(node* x, node* y)
    : x_((expression*)x), y_((expression*)y){ }
  ir::value* codegen(ir::module *) const;

private:
  const expression *x_;
  const expression *y_;
};

class min_expression: public builtin_expression{
public:
  min_expression(node* x, node* y)
    : x_((expression*)x), y_((expression*)y){ }
  ir::value* codegen(ir::module *mod) const;

private:
  const expression *x_;
  const expression *y_;
};

class select_expression: public builtin_expression{
public:
  select_expression(node* pred, node* if_value, node* else_value)
    : pred_((expression*)pred), if_value_((expression*)if_value), else_value_((expression*)else_value) { }
  ir::value* codegen(ir::module *mod) const;

private:
  const expression *pred_;
  const expression *if_value_;
  const expression *else_value_;
};

class trans_expression: public builtin_expression{
public:
  trans_expression(node *arg): arg_(arg) {}
  ir::value* codegen(ir::module *mod) const;

private:
  node* arg_;
};

class sqrt_expression: public builtin_expression{
public:
  sqrt_expression(node *arg): arg_(arg) {}
  ir::value* codegen(ir::module *) const;

private:
  node* arg_;
};

class reduce_expression: public builtin_expression{
public:
  reduce_expression(node *arg): arg_(arg) {}
  ir::value* codegen(ir::module *mod) const;

private:
  node* arg_;
};

class indexing_expression: public postfix_expression{
public:
  indexing_expression(node *lhs, node *slices)
    : lhs_((const expression*)lhs), slices_((const list<slice*>*)slices) {}

  ir::value* codegen(ir::module *) const;

private:
  const expression* lhs_;
  const list<slice*>* slices_;
};



class named_expression: public expression {
public:
  named_expression(node *id): id_((const identifier*)id) { lvalue_ = this; }
  const identifier *id() const { return id_; }
  ir::value* codegen(ir::module * mod) const;

private:
  const identifier *id_;
};

class binary_expression: public expression{
private:
  ir::value* llvm_op(ir::module *mod, ir::builder &bld, ir::value *lhs, ir::value *rhs, const std::string &name) const;

public:
  binary_expression(BIN_OP_T op, node *lhs, node *rhs)
    : op_(op), lhs_((expression*)lhs), rhs_((expression*)rhs) {
  }
  ir::value* codegen(ir::module *) const;

private:
  const BIN_OP_T op_;
  const expression *lhs_;
  const expression *rhs_;
};


class constant: public expression{
public:
  constant(int value): value_(value) { }
  ir::value* codegen(ir::module *mod) const;
  int value() const;

private:
  const int value_;
};

class constant_range: public expression {
public:
  constant_range(node *first, node *last)
    : first_((constant*)first), last_((constant*)last) { }

  ir::value* codegen(ir::module *mod) const;

private:
  constant *first_;
  constant *last_;
};

class string_literal: public expression{
public:
  string_literal(char *&value): value_(value) { }
  ir::value* codegen(ir::module *mod) const;

public:
  std::string value_;
};

class unary_expression: public expression{
private:
  ir::value *llvm_op(ir::builder &builder, ir::value *arg, const std::string &name) const;

public:
  unary_expression(UNARY_OP_T op, node *arg)
      : op_(op),
        arg_((expression*)arg) {
    if(op == DEREF)
      this->lvalue_ = arg_->lvalue();
  }

  UNARY_OP_T get_op() const { return op_; }
  ir::value* codegen(ir::module *mod) const;

private:
  const UNARY_OP_T op_;
  const expression *arg_;
};

class type_name;
class cast_expression: public expression{
private:
  ir::value *llvm_op(ir::builder &builder, ir::type *T, ir::value *arg, const std::string &name) const;

public:
  cast_expression(node *T, node *arg):
    T_((type_name*)T),
    arg_((expression*)arg) { }

  ir::value* codegen(ir::module *mod) const;

public:
  const type_name *T_;
  const expression *arg_;
};

class conditional_expression: public expression{
private:
  ir::value *llvm_op(ir::builder &builder,
                       ir::value *cond, ir::value *true_value, ir::value *false_value,
                       const std::string &name) const;

public:
  conditional_expression(node *cond, node *true_value, node *false_value)
    : cond_((expression*)cond),
      true_value_((expression*)true_value),
      false_value_((expression*)false_value) { }

  ir::value* codegen(ir::module *mod) const;

public:
  const expression *cond_;
  const expression *true_value_;
  const expression *false_value_;
};

class assignment_expression: public expression{
public:
  assignment_expression(node *lvalue, ASSIGN_OP_T op, node *rvalue)
    : lvalue_((named_expression*)lvalue), op_(op), rvalue_((expression*)rvalue) { }

  ir::value* codegen(ir::module *mod) const;
  const expression *lvalue() const { return lvalue_; }
  const expression *rvalue() const { return rvalue_; }

public:
  ASSIGN_OP_T op_;
  const expression *lvalue_;
  const expression *rvalue_;
};


}

}

#endif
