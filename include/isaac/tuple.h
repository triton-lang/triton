/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

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
ISAACAPI expression_tree make_tuple(driver::Context const & context, T const & x, Args... args)
{ return expression_tree(wrap_generic(x), make_tuple(context, args...), op_element(BINARY_TYPE_FAMILY, PAIR_TYPE), context, numeric_type_of(x), {1}); }

inline value_scalar tuple_get(expression_tree::container_type const & tree, size_t root, size_t idx)
{
  for(unsigned int i = 0 ; i < idx ; ++i){
      expression_tree::node node = tree[root];
      if(node.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
        root = node.rhs.node_index;
      else
        return value_scalar(node.rhs.vscalar, node.rhs.dtype);
  }
  return value_scalar(tree[root].lhs.vscalar, tree[root].lhs.dtype);
}



}

#endif
