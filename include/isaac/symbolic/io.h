#ifndef ISAAC_SCHEDULER_IO_H
#define ISAAC_SCHEDULER_IO_H

#include <iostream>
#include "isaac/symbolic/expression.h"

namespace isaac
{

std::string to_string(array_expression_node_subtype const & f);
std::string to_string(lhs_rhs_element const & e);
std::ostream & operator<<(std::ostream & os, array_expression::node const & s_node);
std::string to_string(isaac::array_expression const & s);

}

#endif

