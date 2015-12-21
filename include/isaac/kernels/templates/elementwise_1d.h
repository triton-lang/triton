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

#ifndef ISAAC_BACKEND_TEMPLATES_VAXPY_H
#define ISAAC_BACKEND_TEMPLATES_VAXPY_H

#include "isaac/kernels/templates/base.h"

namespace isaac
{
namespace templates
{

class elementwise_1d_parameters : public base::parameters_type
{
public:
  elementwise_1d_parameters(unsigned int _simd_width, unsigned int _group_size, unsigned int _num_groups, fetching_policy_type _fetching_policy);
  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class elementwise_1d : public base_impl<elementwise_1d, elementwise_1d_parameters>
{
private:
  virtual int is_invalid_impl(driver::Device const &, expression_tree const  &) const;
  std::string generate_impl(std::string const & suffix, expression_tree const  & expressions, driver::Device const & device, mapping_type const & mappings) const;
public:
  elementwise_1d(elementwise_1d::parameters_type const & parameters, binding_policy_t binding_policy = BIND_INDEPENDENT);
  elementwise_1d(unsigned int _simd_width, unsigned int _group_size, unsigned int _num_groups, fetching_policy_type _fetching_policy, binding_policy_t binding_policy = BIND_INDEPENDENT);
  std::vector<int_t> input_sizes(expression_tree const  & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const &);
};

}
}

#endif
