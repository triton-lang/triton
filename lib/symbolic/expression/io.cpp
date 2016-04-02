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

#include <iostream>
#include <sstream>
#include <string>

#include "isaac/symbolic/expression/operations.h"
#include "isaac/symbolic/expression/io.h"
#include "isaac/tools/cpp/string.hpp"
#include "isaac/array.h"

namespace isaac
{

#define ISAAC_MAP_TO_STRING(NAME) case NAME: return #NAME

inline std::string to_string(const op_element& op)
{
  std::string res = to_string(op.type);
  if(op.type_family==REDUCE) res = "reduce<" + res + ">";
  if(op.type_family==REDUCE_ROWS) res = "reduce<" + res + ", rows>";
  if(op.type_family==REDUCE_COLUMNS) res = "reduce<" + res + ", cols>";
  return res;
}

inline std::string to_string(const expression_tree::node &node)
{
  if(node.type==COMPOSITE_OPERATOR_TYPE)
  {
    std::string lhs = tools::to_string(node.binary_operator.lhs);
    std::string op = to_string(node.binary_operator.op);
    std::string rhs = tools::to_string(node.binary_operator.rhs);
    return"node (" + lhs + ", " + op + ", " + rhs + ")";
  }
  switch(node.type)
  {
    case INVALID_SUBTYPE:
      return "empty";
    case VALUE_SCALAR_TYPE:
      return "scalar";
    case DENSE_ARRAY_TYPE:
      return "array";
    default:
      return "unknown";
  }
}

namespace detail
{
  /** @brief Recursive worker routine for printing a whole expression_tree */
  inline void print_node(std::ostream & os, isaac::expression_tree const & s, size_t index, size_t indent = 0)
  {
    expression_tree::data_type const & data = s.data();
    expression_tree::node const & node = data[index];

    for (size_t i=0; i<indent; ++i)
      os << " ";

    os << "Node " << index << ": " << to_string(node) << std::endl;

    if (node.type == COMPOSITE_OPERATOR_TYPE)
    {
      print_node(os, s, node.binary_operator.lhs, indent+1);
      print_node(os, s, node.binary_operator.rhs, indent+1);
    }
  }
}

std::string to_string(isaac::expression_tree const & s)
{
  std::ostringstream os;
  detail::print_node(os, s, s.root());
  return os.str();
}

}

