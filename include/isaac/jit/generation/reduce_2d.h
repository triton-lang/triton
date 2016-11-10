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
#include "isaac/jit/syntax/expression/preset.h"

namespace isaac
{
namespace templates
{

class intelblas_gemv : public external_base
{
private:
  std::string generate_impl(std::string const & suffix, expression_tree const &, driver::Device const & device, symbolic::symbols_table const &) const;
public:
  intelblas_gemv();
  int is_invalid(expression_tree const  &, driver::Device const &) const;
   std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const &, std::string const &, runtime::execution_handler const & h);
  expression_type type() const;
};

class reduce_2d : public parameterized_base
{
protected:
  reduce_2d(unsigned int vwidth, unsigned int ls0, unsigned int ls1, unsigned int ng0, unsigned int ng1, operation_type_family);
private:
  unsigned int lmem_usage(expression_tree const &) const;
  unsigned int temporary_workspace(expression_tree const & expressions) const;
  std::string generate_impl(std::string const & suffix, expression_tree const &, driver::Device const & device, symbolic::symbols_table const &) const;
public:
  virtual std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const &);
  expression_type type() const;
private:
  unsigned int ng0_;
  unsigned int ng1_;
  operation_type_family reduction_type_;
};

class reduce_2d_rows : public reduce_2d
{
public:
  reduce_2d_rows(unsigned int vwidth, unsigned int ls0, unsigned int ls1, unsigned int ng0, unsigned int ng1);
};

class reduce_2d_cols : public reduce_2d
{
public:
  reduce_2d_cols(unsigned int vwidth, unsigned int ls0, unsigned int ls1, unsigned int ng0, unsigned int ng1);
};

}
}

#endif
