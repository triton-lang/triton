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
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/gemm.h"
#include "isaac/jit/generation/base.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace templates
{

base::base()
{}

unsigned int base::lmem_usage(expression_tree const  &) const
{ return 0; }

unsigned int base::registers_usage(expression_tree const  &) const
{ return 0; }

unsigned int base::temporary_workspace(expression_tree const  &) const
{ return 0; }

base::~base()
{ }

std::string base::generate(std::string const & suffix, expression_tree const  & expression, driver::Device const & device)
{
  int err = is_invalid(expression, device);
  if(err != 0)
    throw operation_not_supported_exception("The supplied parameters for this template are invalid : err " + tools::to_string(err));

  //Create mapping
  symbolic::symbols_table mapping = symbolic::symbolize(expression);
  return generate_impl(suffix, expression, device, mapping);
}


/* External base */
external_base::external_base()
{ }

std::string external_base::generate_impl(std::string const &, expression_tree const &, driver::Device const &, symbolic::symbols_table const &) const
{ return ""; }

unsigned int external_base::temporary_workspace(expression_tree const &) const
{ return 0; }

unsigned int external_base::lmem_usage(expression_tree const &) const
{ return 0; }

unsigned int external_base::registers_usage(expression_tree const &) const
{ return 0; }

/* Parameterized base */
int parameterized_base::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{ return TEMPLATE_VALID; }

parameterized_base::parameterized_base(unsigned int vwidth, int_t ls0, int_t ls1): vwidth_(vwidth), ls0_(ls0), ls1_(ls1)
{ }

unsigned int parameterized_base::ls0() const
{ return ls0_; }

unsigned int parameterized_base::ls1() const
{ return ls1_; }

int parameterized_base::is_invalid(expression_tree const  & expressions, driver::Device const & device) const
{
  //Query device informations
  size_t lmem_available = device.local_mem_size();
  size_t lmem_used = lmem_usage(expressions);
  if (lmem_used>lmem_available)
    return TEMPLATE_LOCAL_MEMORY_OVERFLOW;

  //Invalid work group size
  size_t max_workgroup_size = device.max_work_group_size();
  std::vector<size_t> max_work_item_sizes = device.max_work_item_sizes();
  if (ls0_*ls1_ > max_workgroup_size)
    return TEMPLATE_WORK_GROUP_SIZE_OVERFLOW;
  if (ls0_ > max_work_item_sizes[0])
    return TEMPLATE_LOCAL_SIZE_0_OVERFLOW;

  if (ls1_ > max_work_item_sizes[1])
    return TEMPLATE_LOCAL_SIZE_1_OVERFLOW;

  //Invalid SIMD Width
  if (vwidth_!=1 && vwidth_!=2 && vwidth_!=3 && vwidth_!=4)
    return TEMPLATE_INVALID_SIMD_WIDTH;

  return is_invalid_impl(device, expressions);
}

std::shared_ptr<base> base::getptr()
{ return shared_from_this(); }

}
}
