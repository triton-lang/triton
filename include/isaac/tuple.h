#ifndef ISAAC_TUPLE_H
#define ISAAC_TUPLE_H

#include "isaac/common/numeric_type.h"

#include "isaac/defines.h"
#include "isaac/value_scalar.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

template<class T> typename std::enable_if<!std::is_arithmetic<T>::value, T const &>::type wrap_generic(T const & x){ return x;}
template<class T> typename std::enable_if<std::is_arithmetic<T>::value, value_scalar>::type wrap_generic(T x) { return value_scalar(x); }

template<typename T>
ISAACAPI typename std::conditional<std::is_arithmetic<T>::value, value_scalar, T const &>::type make_tuple(driver::Context const &, T const & x)
{ return wrap_generic(x); }

template<typename T, typename... Args>
ISAACAPI math_expression make_tuple(driver::Context const & context, T const & x, Args... args)
{ return math_expression(wrap_generic(x), make_tuple(context, args...), op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_PAIR_TYPE), context, numeric_type_of(x), size4(1)); }

inline value_scalar tuple_get(math_expression::container_type const & tree, size_t root, size_t idx)
{
  for(unsigned int i = 0 ; i < idx ; ++i){
      math_expression::node node = tree[root];
      if(node.rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
        root = node.rhs.node_index;
      else
        return value_scalar(node.rhs.vscalar, node.rhs.dtype);
  }
  return value_scalar(tree[root].lhs.vscalar, tree[root].lhs.dtype);
}



}

#endif
