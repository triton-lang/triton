#include "parser.hpp"
#include <cassert>
#include <list>
#include <string>

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
class node { };

struct token: public node{
  token(ASSIGN_OP_T value): assign_op(value){ }
  token(BIN_OP_T value): bin_op(value){ }
  token(UNARY_OP_T value): unary_op(value){ }
  token(TYPE_T value): type(value){ }

  union {
    ASSIGN_OP_T assign_op;
    BIN_OP_T bin_op;
    UNARY_OP_T unary_op;
    TYPE_T type;
  };
};

template<class T>
class list: public node {
public:
  list(const T& x): values_{x} {}
  node* append(const T& x) { values_.push_back(x); return this;}

private:
  std::list<T> values_;
};

template<class T>
node* append_ptr_list(node *result, node *in){
  return static_cast<list<T*>*>(result)->append((T*)in);
}

class binary_operator: public node{
public:
  binary_operator(node *op, node *lhs, node *rhs)
    : op_(((token*)op)->bin_op), lhs_(lhs), rhs_(rhs) { }
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

class identifier: public node{
public:
  identifier(char *&name): name_(name) { }

private:
  std::string name_;
};

class string_literal: public node{
public:
  string_literal(char *&value): value_(value) { }

public:
  std::string value_;
};

class unary_operator: public node{
public:
  unary_operator(node *op, node *arg)
    : op_(((token*)op)->unary_op), arg_(arg) { }
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
  assignment_expression(node *lvalue, node *op, node *rvalue)
    : lvalue_(lvalue), op_(((token*)op)->assign_op), rvalue_(rvalue) { }

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
class declarator: public node{

};

class pointer_declarator: public declarator{
public:
  pointer_declarator(unsigned order)
    : order_(order) { }

  pointer_declarator *inc(){
    order_ += 1;
    return this;
  }

private:
  unsigned order_;
};

class tile_declarator: public declarator{
public:
  tile_declarator(node *shapes)
    : shapes_((list<constant*>*)(shapes)) { }

public:
  const list<constant*>* shapes_;
};

class parameter: public declarator {
public:
  parameter(node *spec, node *decl)
    : spec_(((token*)spec)->type), decl_(decl) { }

public:
  const TYPE_T spec_;
  const node *decl_;
};

class function_declarator: public declarator{
public:
  function_declarator(node *args)
    : args_((list<node*>)args) { }

public:
  const list<node*> args_;
};

class compound_declarator: public declarator{
public:
  compound_declarator(node *ptr, node *tile)
    : ptr_(ptr), tile_(tile) { }

public:
  const node *ptr_;
  const node *tile_;
};

class init_declarator : public declarator{
public:
  init_declarator(node *decl, node *initializer)
  : decl_(decl), initializer_(initializer){ }

public:
  const node *decl_;
  const node *initializer_;
};


class type: public node{
public:
  type(node *spec, node * decl)
    : spec_(((token*)spec)->type), decl_(decl) { }

public:
  const TYPE_T spec_;
  const node *decl_;
};

class translation_unit: public node{
public:
  translation_unit(node *item)
    : decls_(item) { }

  translation_unit *add(node *item) {
    decls_.append(item);
    return this;
  }

private:
  list<node*> decls_;
};

class function_definition: public node{
public:
  function_definition(node *header, node *body)
    : header_((declarator *)header), body_((compound_statement*)body) { }

public:
  const declarator *header_;
  const compound_statement *body_;
};

}
