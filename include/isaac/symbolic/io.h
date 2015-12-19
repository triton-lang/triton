#ifndef ISAAC_SCHEDULER_IO_H
#define ISAAC_SCHEDULER_IO_H

#include <iostream>
#include "isaac/symbolic/expression.h"

namespace isaac
{

std::string to_string(node_type const & f);
std::string to_string(lhs_rhs_element const & e);
std::ostream & operator<<(std::ostream & os, math_expression::node const & s_node);
std::string to_string(isaac::math_expression const & s);

}

#endif

