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

#ifndef ISAAC_BACKEND_TEMPLATES_MAXPY_H
#define ISAAC_BACKEND_TEMPLATES_MAXPY_H

#include <vector>
#include "isaac/jit/generation/base.h"

namespace isaac
{
namespace templates
{

class elementwise_2d : public parameterized_base
{
private:
  int is_invalid_impl(driver::Device const &, expression_tree const  &) const;
  std::string generate_impl(std::string const & suffix, expression_tree const  & expressions, driver::Device const & device, symbolic::symbols_table const & mapping) const;
public:
  elementwise_2d(unsigned int vwidth, unsigned int ls0, unsigned int ls1,  unsigned int ng0, unsigned int ng1);
  std::vector<int_t> input_sizes(expression_tree const  & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const &);
  expression_type type() const;
private:
  unsigned int ng0_;
  unsigned int ng1_;
};

}
}

#endif
