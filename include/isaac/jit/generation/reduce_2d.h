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

#ifndef ISAAC_BACKEND_TEMPLATES_MDOT_H
#define ISAAC_BACKEND_TEMPLATES_MDOT_H

#include <vector>

#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/jit/generation/base.h"

namespace isaac
{
namespace templates
{
struct reduce_2d_parameters : public base::parameters_type
{
  reduce_2d_parameters(uint32_t _vwidth,
                                uint32_t _ls0, uint32_t _ls1,
                                uint32_t _ng0, uint32_t _ng1, fetch_type _fetch_policy);
  uint32_t ng0;
  uint32_t ng1;
  fetch_type fetch_policy;
};


class reduce_2d : public base_impl<reduce_2d, reduce_2d_parameters>
{
protected:
  reduce_2d(reduce_2d::parameters_type const & , operation_type_family);
private:
  int is_invalid_impl(driver::Device const &, expression_tree const &) const;
  uint32_t lmem_usage(expression_tree const &) const;
  uint32_t temporary_workspace(expression_tree const & expressions) const;
  std::string generate_impl(std::string const & suffix, expression_tree const &, driver::Device const & device, symbolic::symbols_table const &) const;
public:
  virtual std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const &);
private:
  operation_type_family reduction_type_;
};

class reduce_2d_rows : public reduce_2d
{
public:
  reduce_2d_rows(reduce_2d::parameters_type  const &);
  reduce_2d_rows(uint32_t simd, uint32_t ls1, uint32_t ls2, uint32_t ng1, uint32_t ng2, fetch_type fetch);
};

class reduce_2d_cols : public reduce_2d
{
public:
  reduce_2d_cols(reduce_2d::parameters_type  const &);
  reduce_2d_cols(uint32_t simd, uint32_t ls1, uint32_t ls2, uint32_t ng1, uint32_t ng2, fetch_type fetch);
};

}
}

#endif
