#include "parser.hpp"
#include <cassert>
#include <list>
#include <string>

typedef yytokentype token_type;

namespace ast{

class node { };

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
  enum OP_T{
    MUL, DIV, REM,
    ADD, SUB,
    LEFT_SHIFT, RIGHT_SHIFT,
    LT, GT,
    LE, GE,
    EQ, NE,
    AND, XOR, OR,
    LAND, LOR
  };

  static OP_T get_op(token_type token){
    switch(token){
    case LEFT_OP: return LEFT_SHIFT;
    case RIGHT_OP: return RIGHT_SHIFT;
    case LE_OP: return LE;
    case GE_OP: return GE;
    case EQ_OP: return EQ;
    case NE_OP: return NE;
    case AND_OP: return LAND;
    case OR_OP: return LOR;
    default: assert(false && "unreachable"); throw;
    }
  }

  static OP_T get_op(char token){
    switch(token){
    case '*': return MUL;
    case '/': return DIV;
    case '%': return REM;
    case '+': return ADD;
    case '-': return SUB;
    case '<': return LT;
    case '>': return GT;
    case '&': return AND;
    case '^': return XOR;
    case '|': return OR;
    default: assert(false && "unreachable"); throw;
    }
  }

public:
  binary_operator(token_type op, node *lhs, node *rhs)
    : op_(get_op(op)), lhs_(lhs), rhs_(rhs) { }
  binary_operator(char op, node *lhs, node *rhs)
    : op_(get_op(op)), lhs_(lhs), rhs_(rhs){ }

private:
  const OP_T op_;
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
  unary_operator(token_type token, node *arg): token_(token), arg_(arg) { }

private:
  const token_type token_;
  const node *arg_;
};

class cast_operator: public node{
public:
  cast_operator(token_type type, node *arg): type_(type), arg_(arg) { }

public:
  const token_type type_;
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
  typedef binary_operator::OP_T op_t;

public:
  assignment_expression(node *lvalue, token_type op, node *rvalue)
    : lvalue_(lvalue), op_(binary_operator::get_op(op)), rvalue_(rvalue) { }

public:
  op_t op_;
  const node *lvalue_;
  const node *rvalue_;
};

class compound_statement: public node{
public:
  compound_statement() : statements_() {}
  compound_statement(node *stmt): statements_{stmt} {}
  compound_statement* append(node *stmt) { statements_.push_back(stmt); return this; }

private:
  std::list<node*> statements_;
};

class selection_statement: public node{
public:
  selection_statement(node *cond, node *if_value, node *else_value = nullptr)
    : cond_(cond), if_value_(if_value), else_value_(else_value) { }

public:
  const node *cond_;
  const node *if_value_;
  const node *else_value_;
};

class iteration_statement: public node{
public:
  iteration_statement(node *init, node *stop, node *exec, node *statements)
    : init_(init), stop_(stop), exec_(exec), statements_(statements) { }

private:
  const node *init_;
  const node *stop_;
  const node *exec_;
  const node *statements_;
};

class no_op: public node { };

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
  parameter(token_type type, node *decl)
    : type_(type), decl_(decl) { }

public:
  const token_type type_;
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

class declaration: public node{
public:
  declaration(node *spec, node *init)
    : spec_(spec), init_(init) { }

public:
  const node *spec_;
  const node *init_;
};

class type: public node{
public:
  type(token_type spec, node * decl)
    : spec_(spec), decl_(decl) { }

public:
  const token_type spec_;
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
