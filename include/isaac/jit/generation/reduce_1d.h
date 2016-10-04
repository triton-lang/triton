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

#ifndef ISAAC_BACKEND_TEMPLATES_DOT_H
#define ISAAC_BACKEND_TEMPLATES_DOT_H

#include "isaac/jit/generation/base.h"

namespace isaac
{
namespace templates
{

class reduce_1d : public parameterized_base
{
private:
  unsigned int lmem_usage(expression_tree const  & expressions) const;
  int is_invalid_impl(driver::Device const &, expression_tree const  &) const;
  unsigned int temporary_workspace(expression_tree const & expressions) const;
  inline void reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<symbolic::reduce_1d*> exprs,
                                     std::string const & buf_str, std::string const & buf_value_str, driver::backend_type backend) const;
  std::string generate_impl(std::string const & suffix,  expression_tree const  & expressions, driver::Device const & device, symbolic::symbols_table const & mapping) const;

public:
  reduce_1d(unsigned int vwidth, unsigned int ls, unsigned int ng, fetch_type fetch);
  std::vector<int_t> input_sizes(expression_tree const  & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const &);
  expression_type type() const;

private:
  unsigned int ng_;
  fetch_type fetch_;
  std::vector< driver::Buffer > tmp_;
  std::vector< driver::Buffer > tmpidx_;
};

}
}

#endif
