#ifndef ISAAC_SCHEDULER_IO_H
#define ISAAC_SCHEDULER_IO_H

#include <iostream>
#include "isaac/symbolic/expression.h"

namespace isaac
{

std::string to_string(node_type const & f);
std::string to_string(tree_node const & e);
std::ostream & operator<<(std::ostream & os, expression_tree::node const & s_node);
std::string to_string(isaac::expression_tree const & s);

}

#endif

