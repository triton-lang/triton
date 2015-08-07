#include <iostream>
#include <sstream>

#include "isaac/symbolic/io.h"

#include <string>

namespace isaac
{

#define ISAAC_MAP_TO_STRING(NAME) case NAME: return #NAME

inline std::string to_string(array_expression_node_subtype const & f)
{
  switch(f)
  {
    ISAAC_MAP_TO_STRING(INVALID_SUBTYPE);
    ISAAC_MAP_TO_STRING(VALUE_SCALAR_TYPE);
    ISAAC_MAP_TO_STRING(DENSE_ARRAY_TYPE);
    default: return "UNKNOWN";
  }
}

inline std::string to_string(lhs_rhs_element const & e)
{
  if(e.type_family==COMPOSITE_OPERATOR_FAMILY)
  {
    return"COMPOSITE [" + std::to_string(e.node_index) + "]";
  }
  return std::to_string(e.subtype);
}

inline std::ostream & operator<<(std::ostream & os, array_expression::node const & s_node)
{
  os << "LHS: " << to_string(s_node.lhs) << "|" << s_node.lhs.dtype << ", "
     << "OP: "  << s_node.op.type_family << " | " << s_node.op.type << ", "
     << "RHS: " << to_string(s_node.rhs) << "|" << s_node.rhs.dtype;

  return os;
}


namespace detail
{
  /** @brief Recursive worker routine for printing a whole array_expression */
  inline void print_node(std::ostream & os, isaac::array_expression const & s, size_t node_index, size_t indent = 0)
  {
    array_expression::container_type const & nodes = s.tree();
    array_expression::node const & current_node = nodes[node_index];

    for (size_t i=0; i<indent; ++i)
      os << " ";

    os << "Node " << node_index << ": " << current_node << std::endl;

    if (current_node.lhs.type_family == COMPOSITE_OPERATOR_FAMILY)
      print_node(os, s, current_node.lhs.node_index, indent+1);

    if (current_node.rhs.type_family == COMPOSITE_OPERATOR_FAMILY)
      print_node(os, s, current_node.rhs.node_index, indent+1);
  }
}

std::string to_string(isaac::array_expression const & s)
{
  std::ostringstream os;
  detail::print_node(os, s, s.root());
  return os.str();
}

}

