#ifndef TRITON_INCLUDE_LANG_NODE_H
#define TRITON_INCLUDE_LANG_NODE_H

#include <vector>
#include "ops.h"

namespace triton{


namespace ir{
  class function;
  class value;
  class type;
  class builder;
  class module;
}

namespace lang{

class expression;
class pointer;
class identifier;
class constant;
class compound_statement;
class initializer;
class declaration_specifier;
class function;

// Node
class node {
protected:
  static ir::value* explicit_cast(ir::builder &builder, ir::value *src, ir::type *dst_ty);
  static void implicit_broadcast(ir::module *mod, ir::type *dst_ty, ir::value *&src);
  static void implicit_broadcast(ir::module *mod, ir::value *&lhs, ir::value *&rhs);
  static void implicit_cast(ir::builder &builder, ir::value *&lhs, ir::value *&rhs,
                            bool &is_float, bool &is_ptr, bool &is_int, bool &is_signed);
public:
  virtual ir::value* codegen(ir::module *) const { return nullptr; }
};

class block_item: public node{
};

template<class T>
class list: public node {
public:
  list(const T& x): values_(1, x) {}

  node* append(const T& x){
    values_.push_back(x);
    return this;
  }

  ir::value* codegen(ir::module * mod) const{
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

}

}

#endif
