#include <iostream>
#include <sstream>
#include <string>

#include "isaac/symbolic/io.h"
#include "to_string.hpp"

namespace isaac
{

#define ISAAC_MAP_TO_STRING(NAME) case NAME: return #NAME

inline std::string to_string(node_type const & f)
{
  switch(f)
  {
    ISAAC_MAP_TO_STRING(INVALID_SUBTYPE);
    ISAAC_MAP_TO_STRING(VALUE_SCALAR_TYPE);
    ISAAC_MAP_TO_STRING(DENSE_ARRAY_TYPE);
    default: return "UNKNOWN";
  }
}

inline std::string to_string(tree_node const & e)
{
  if(e.subtype==COMPOSITE_OPERATOR_TYPE)
  {
    return"COMPOSITE [" + tools::to_string(e.node_index) + "]";
  }
  return tools::to_string(e.subtype);
}

inline std::ostream & operator<<(std::ostream & os, expression_tree::node const & s_node)
{
  os << "LHS: " << to_string(s_node.lhs) << "|" << s_node.lhs.dtype << ", "
     << "OP: "  << s_node.op.type_family << " | " << s_node.op.type << ", "
     << "RHS: " << to_string(s_node.rhs) << "|" << s_node.rhs.dtype;

  return os;
}


namespace detail
{
  /** @brief Recursive worker routine for printing a whole expression_tree */
  inline void print_node(std::ostream & os, isaac::expression_tree const & s, size_t node_index, size_t indent = 0)
  {
    expression_tree::container_type const & nodes = s.tree();
    expression_tree::node const & current_node = nodes[node_index];

    for (size_t i=0; i<indent; ++i)
      os << " ";

    os << "Node " << node_index << ": " << current_node << std::endl;

    if (current_node.lhs.subtype == COMPOSITE_OPERATOR_TYPE)
      print_node(os, s, current_node.lhs.node_index, indent+1);

    if (current_node.rhs.subtype == COMPOSITE_OPERATOR_TYPE)
      print_node(os, s, current_node.rhs.node_index, indent+1);
  }
}

std::string to_string(isaac::expression_tree const & s)
{
  std::ostringstream os;
  detail::print_node(os, s, s.root());
  return os.str();
}

}

