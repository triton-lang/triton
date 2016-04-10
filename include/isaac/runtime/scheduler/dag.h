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

#ifndef _ISAAC_SYMBOLIC_SCHEDULER_DAG_H
#define _ISAAC_SYMBOLIC_SCHEDULER_DAG_H

#include <map>
#include <vector>
#include <memory>
#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/runtime/scheduler/strategies/strategy.h"

namespace isaac
{
namespace runtime
{
namespace scheduler
{

class dag
{
private:
  static int_t last_index(array_base* x);
  static bool overlap(expression_tree::node const & x, expression_tree::node const & y);
  static tuple repack(int_t start, const tuple &ld);

public:
  dag();
  void append(expression_tree const & job, std::string const & name = "");
  array_base& create_temporary(array_base* tmp);
  void export_graphviz(std::string const & path);
  adjacency_t const & adjacency() const;

private:
  adjacency_t adjacency_;
  std::vector<expression_tree> jobs_;
  std::vector<std::string> names_;
  std::vector<std::shared_ptr<array_base> > tmp_;
};

}
}
}

#endif
