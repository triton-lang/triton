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

#ifndef ISAAC_SCHEDULER_IO_H
#define ISAAC_SCHEDULER_IO_H

#include <iostream>
#include "isaac/symbolic/expression/expression.h"

namespace isaac
{

std::string to_string(node_type const & f);
std::string to_string(expression_tree::node const & e);
std::ostream & operator<<(std::ostream & os, expression_tree::node const & s_node);
std::string to_string(isaac::expression_tree const & s);

}

#endif

