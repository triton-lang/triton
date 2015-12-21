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

#include <cassert>
#include <algorithm>
#include <string>

#include "isaac/array.h"
#include "isaac/tuple.h"
#include "isaac/kernels/keywords.h"
#include "isaac/kernels/templates/elementwise_1d.h"
#include "isaac/kernels/templates/reduce_1d.h"
#include "isaac/kernels/templates/elementwise_2d.h"
#include "isaac/kernels/templates/reduce_2d.h"
#include "isaac/kernels/templates/matrix_product.h"
#include "isaac/kernels/templates/base.h"
#include "isaac/kernels/parse.h"
#include "isaac/exception/unknown_datatype.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/symbolic/io.h"

#include "tools/map.hpp"
#include "cpp/to_string.hpp"

namespace isaac
{
namespace templates
{

base::parameters_type::parameters_type(unsigned int _simd_width, int_t _local_size_1, int_t _local_size_2, int_t _num_kernels) : simd_width(_simd_width), local_size_0(_local_size_1), local_size_1(_local_size_2), num_kernels(_num_kernels)
{ }


bool base::requires_fallback(expression_tree const  & expression)
{
  for(expression_tree::node const & node: expression.tree())
    if(  (node.lhs.subtype==DENSE_ARRAY_TYPE && (node.lhs.array->stride()[0]>1 || node.lhs.array->start()>0))
      || (node.rhs.subtype==DENSE_ARRAY_TYPE && (node.rhs.array->stride()[0]>1 || node.rhs.array->start()>0)))
      return true;
  return false;
}

int_t base::vector_size(expression_tree::node const & node)
{
  if (node.op.type==MATRIX_DIAG_TYPE)
    return std::min<int_t>(node.lhs.array->shape()[0], node.lhs.array->shape()[1]);
  else if (node.op.type==MATRIX_ROW_TYPE)
    return node.lhs.array->shape()[1];
  else if (node.op.type==MATRIX_COLUMN_TYPE)
    return node.lhs.array->shape()[0];
  else
    return node.lhs.array->shape().max();

}

std::pair<int_t, int_t> base::matrix_size(expression_tree::container_type const & tree, expression_tree::node const & node)
{
  if (node.op.type==VDIAG_TYPE)
  {
    int_t size = node.lhs.array->shape()[0];
    return std::make_pair(size,size);
  }
  else if(node.op.type==REPEAT_TYPE)
  {
    size_t rep0 = tuple_get(tree, node.rhs.node_index, 0);
    size_t rep1 = tuple_get(tree, node.rhs.node_index, 1);
    std::cout << rep0 << " " << rep1 << std::endl;
    return std::make_pair(node.lhs.array->shape()[0]*rep0, node.lhs.array->shape()[1]*rep1);
  }
  else
    return std::make_pair(node.lhs.array->shape()[0],node.lhs.array->shape()[1]);
}


base::base(binding_policy_t binding_policy) : binding_policy_(binding_policy)
{}

unsigned int base::lmem_usage(expression_tree const  &) const
{ return 0; }

unsigned int base::registers_usage(expression_tree const  &) const
{ return 0; }

unsigned int base::temporary_workspace(expression_tree const  &) const
{ return 0; }

base::~base()
{
}

std::string base::generate(std::string const & suffix, expression_tree const  & expression, driver::Device const & device)
{
  int err = is_invalid(expression, device);
  if(err != 0)
    throw operation_not_supported_exception("The supplied parameters for this template are invalid : err " + tools::to_string(err));

  //Create mapping
  mapping_type mapping;
  std::unique_ptr<symbolic_binder> binder;
  if (binding_policy_==BIND_SEQUENTIAL)
      binder.reset(new bind_sequential());
  else
      binder.reset(new bind_independent());

  traverse(expression, expression.root(), map_functor(*binder, mapping, device), true);
  return generate_impl(suffix, expression, device, mapping);
}

template<class TType, class PType>
int base_impl<TType, PType>::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{ return TEMPLATE_VALID; }

template<class TType, class PType>
base_impl<TType, PType>::base_impl(parameters_type const & parameters, binding_policy_t binding_policy) : base(binding_policy), p_(parameters)
{ }

template<class TType, class PType>
unsigned int base_impl<TType, PType>::local_size_0() const
{ return p_.local_size_0; }

template<class TType, class PType>
unsigned int base_impl<TType, PType>::local_size_1() const
{ return p_.local_size_1; }

template<class TType, class PType>
std::shared_ptr<base> base_impl<TType, PType>::clone() const
{ return std::shared_ptr<base>(new TType(*dynamic_cast<TType const *>(this))); }

template<class TType, class PType>
int base_impl<TType, PType>::is_invalid(expression_tree const  & expressions, driver::Device const & device) const
{
  //Query device informations
  size_t lmem_available = device.local_mem_size();
  size_t lmem_used = lmem_usage(expressions);
  if (lmem_used>lmem_available)
    return TEMPLATE_LOCAL_MEMORY_OVERFLOW;

  //Invalid work group size
  size_t max_workgroup_size = device.max_work_group_size();
  std::vector<size_t> max_work_item_sizes = device.max_work_item_sizes();
  if (p_.local_size_0*p_.local_size_1 > max_workgroup_size)
    return TEMPLATE_WORK_GROUP_SIZE_OVERFLOW;
  if (p_.local_size_0 > max_work_item_sizes[0])
    return TEMPLATE_LOCAL_SIZE_0_OVERFLOW;

  if (p_.local_size_1 > max_work_item_sizes[1])
    return TEMPLATE_LOCAL_SIZE_1_OVERFLOW;

  //Invalid SIMD Width
  if (p_.simd_width!=1 && p_.simd_width!=2 && p_.simd_width!=3 && p_.simd_width!=4)
    return TEMPLATE_INVALID_SIMD_WIDTH;

  return is_invalid_impl(device, expressions);
}

template class base_impl<elementwise_1d, elementwise_1d_parameters>;
template class base_impl<reduce_1d, reduce_1d_parameters>;
template class base_impl<elementwise_2d, elementwise_2d_parameters>;
template class base_impl<reduce_2d, reduce_2d_parameters>;
template class base_impl<matrix_product, matrix_product_parameters>;

}
}
