#ifndef TRITON_INCLUDE_LANG_STATEMENT_H
#define TRITON_INCLUDE_LANG_STATEMENT_H

#include "expression.h"

namespace triton{


namespace ir{
  class function;
  class value;
  class type;
  class builder;
  class module;
}

namespace lang{

class declaration;

class statement: public block_item{
};

// Expression
class expression_statement: public statement{
public:
  expression_statement(node *expr, node *mask = nullptr)
    : expr_((expression*)expr), pred_((expression*)mask){ }

  ir::value* codegen(ir::module * mod) const;

private:
  expression *expr_;
  expression *pred_;
};

// Compound
class compound_statement: public statement{
  typedef list<declaration*>* declarations_t;
  typedef list<statement*>* statements_t;

public:
  compound_statement(node* items)
    : items_((list<block_item*>*)items){}

  ir::value* codegen(ir::module * mod) const;

private:
  list<block_item*>* items_;
};

// Selection
class selection_statement: public statement{
public:
  selection_statement(node *cond, node *if_value, node *else_value = nullptr)
    : cond_(cond), then_value_(if_value), else_value_(else_value) { }

  ir::value* codegen(ir::module *mod) const;

public:
  const node *cond_;
  const node *then_value_;
  const node *else_value_;
};

// Iteration
class iteration_statement: public statement{
public:
  iteration_statement(node *init, node *stop, node *exec, node *statements)
    : init_(init), stop_(stop), exec_(exec), statements_(statements)
  { }

  ir::value* codegen(ir::module *mod) const;

private:
  const node *init_;
  const node *stop_;
  const node *exec_;
  const node *statements_;
};

// While
class while_statement: public statement{
public:
  while_statement(node *cond, node *statements)
    : cond_(cond), statements_(statements)
  { }

  ir::value* codegen(ir::module *) const;

private:
  const node *cond_;
  const node *statements_;
};

// Jump
class jump_statement: public statement{
public:
  using statement::statement;
};

// Continue
class continue_statement: public jump_statement{
public:
  ir::value* codegen(ir::module *mod) const;
};

// No op
class no_op: public statement { };

}

}

#endif
