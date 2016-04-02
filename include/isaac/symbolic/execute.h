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

#ifndef _ISAAC_SYMBOLIC_EXECUTE_H
#define _ISAAC_SYMBOLIC_EXECUTE_H

#include "isaac/profiles/profiles.h"
#include "isaac/symbolic/expression/expression.h"

namespace isaac
{

namespace symbolic
{

namespace detail
{
  typedef std::vector<std::pair<size_t, expression_type> > breakpoints_t;
  expression_type parse(expression_tree const & tree, breakpoints_t & bp);
  expression_type parse(expression_tree const & tree, size_t idx, breakpoints_t & bp);
}

/** @brief Executes a expression_tree on the given queue for the given models map*/
void execute(execution_handler const & , profiles::map_type &);

/** @brief Executes a expression_tree on the default models map*/
void execute(execution_handler const &);

}

}

#endif
