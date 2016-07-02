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

#include <cstring>
#include <iostream>
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/syntax/engine/process.h"
#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/vector_types.hpp"


namespace isaac
{
namespace templates
{

elementwise_2d_parameters::elementwise_2d_parameters(unsigned int _vwidth,
                          unsigned int _ls0, unsigned int _ls1,
                          unsigned int _num_groups_0, unsigned int _num_groups_1,
                          fetch_type _fetch) : base::parameters_type(_vwidth, _ls0, _ls1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetch(_fetch){ }



int elementwise_2d::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{
  if (p_.vwidth>1)
    return TEMPLATE_INVALID_SIMD_WIDTH;
  if(p_.fetch==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string elementwise_2d::generate_impl(std::string const & suffix, expression_tree const  & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::vector<std::size_t> assigned = symbolic::find(tree, [&](expression_tree::node const & node){return node.type==COMPOSITE_OPERATOR_TYPE && is_assignment(node.binary_operator.op.type);});
  std::vector<std::size_t> assigned_left;
  std::vector<std::size_t> assigned_right;
  for(std::size_t idx: assigned){
    assigned_left.push_back(tree[idx].binary_operator.lhs);
    assigned_right.push_back(tree[idx].binary_operator.rhs);
  }
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.ls0 << "," << p_.ls1 << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void elementwise_2d" << suffix << "($SIZE_T M, $SIZE_T N, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();


  fetching_loop_info(p_.fetch, "M", stream, init0, upper_bound0, inc0,  "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", device);
  stream << "for($SIZE_T i = " << init0 << "; i < " << upper_bound0 << "; i += " << inc0 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  fetching_loop_info(p_.fetch, "N", stream, init1, upper_bound1, inc1, "$GLOBAL_IDX_1", "$GLOBAL_SIZE_1", device);
  stream << "for($SIZE_T j = " << init1 << "; j < " << upper_bound1 << "; j += " << inc1 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  //Declares register to store results
  for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_left, false))
    stream << sym->process("#scalartype #name;") << std::endl;

  //Load to registers
  for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_right, false))
    stream << sym->process("#scalartype #name = at(i, j);") << std::endl;

  for(std::size_t idx: assigned)
    stream << symbols.at(idx)->evaluate({{"leaf", "#name"}}) << ";" << std::endl;

  //Writes back
  for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_left, false))
    stream << sym->process("at(i, j) = #name;") << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

elementwise_2d::elementwise_2d(parameters_type const & parameters, fusion_policy_t fusion_policy) :
  base_impl<elementwise_2d, elementwise_2d_parameters>(parameters, fusion_policy){ }

elementwise_2d::elementwise_2d(unsigned int simd, unsigned int ls1, unsigned int ls2,
                               unsigned int ng1, unsigned int ng2, fetch_type fetch,
                               fusion_policy_t bind):
    base_impl<elementwise_2d, elementwise_2d_parameters>(elementwise_2d_parameters(simd, ls1, ls2, ng1, ng2, fetch), bind)
{}

std::vector<int_t> elementwise_2d::input_sizes(expression_tree const  & expression) const{
  return expression.shape();
}

void elementwise_2d::enqueue(driver::CommandQueue & /*queue*/, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const  & expressions = control.x();
  std::string name = "elementwise_2d";
  name +=suffix;
  driver::Kernel kernel(program, name.c_str());
  driver::NDRange global(p_.ls0*p_.num_groups_0, p_.ls1*p_.num_groups_1);
  driver::NDRange local(p_.ls0, p_.ls1);
  unsigned int current_arg = 0;
  std::vector<int_t> MN = input_sizes(expressions);
  kernel.setSizeArg(current_arg++, MN[0]);
  kernel.setSizeArg(current_arg++, MN[1]);
  symbolic::set_arguments(expressions, kernel, current_arg, fusion_policy_);

  control.execution_options().enqueue(program.context(), kernel, global, local);
}

}
}
