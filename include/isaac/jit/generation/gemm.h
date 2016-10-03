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

#ifndef ISAAC_BACKEND_TEMPLATES_MPRODUCT_H
#define ISAAC_BACKEND_TEMPLATES_MPRODUCT_H

#include "isaac/jit/generation/base.h"
#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/jit/syntax/expression/preset.h"

namespace isaac
{
namespace templates
{

class gemm : public base_impl
{
private:
  uint32_t temporary_workspace(expression_tree const & expressions) const;
  uint32_t lmem_usage(expression_tree const & expressions) const;
  uint32_t registers_usage(expression_tree const & expressions) const;
  int is_invalid_impl(driver::Device const &, expression_tree const &) const;
  std::string generate_impl(std::string const & suffix, expression_tree const & expressions, driver::Device const & device, symbolic::symbols_table const &) const;
  void enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K, const expression_tree::node &A, const expression_tree::node &B, const expression_tree::node &C,
                     value_scalar const &alpha, value_scalar const &beta, driver::Program const & program, std::string const & suffix, runtime::execution_options_type const & options);
  std::vector<int_t> infos(expression_tree const & expressions,  isaac::symbolic::preset::gemm::args &arguments) const;
public:
  gemm(uint32_t simd, int_t ls0, int_t KL, int_t ls1, int_t D
       , int_t ms, int_t ks, int_t ns, fetch_type Afetch , fetch_type Bfetch
       , int_t lf0, int_t lf1, char A_trans, char B_trans);
  std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const &ctr);
private:
  //Parameters
  uint32_t kL_;
  uint32_t depth_;

  uint32_t mS_;
  uint32_t kS_;
  uint32_t nS_;

  fetch_type Afetch_;
  fetch_type Bfetch_;

  uint32_t lf0_;
  uint32_t lf1_;

  uint32_t mL_;
  uint32_t nL_;

  bool prefetch_;
  bool unroll_outer_;
  //
  const char A_trans_;
  const char B_trans_;
  expression_type type_;
};

class gemm_nn : public gemm
{
public:
  gemm_nn(uint32_t vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetch_type Afetch , fetch_type Bfetch
                      , int_t lf0, int_t lf1);
};

class gemm_tn : public gemm
{
public:
  gemm_tn(uint32_t vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetch_type Afetch , fetch_type Bfetch
                      , int_t lf0, int_t lf1);
};


class gemm_nt : public gemm
{
public:
  gemm_nt(uint32_t vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetch_type Afetch , fetch_type Bfetch
                      , int_t lf0, int_t lf1);
};


class gemm_tt : public gemm
{
public:
  gemm_tt(uint32_t vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetch_type Afetch , fetch_type Bfetch
                      , int_t lf0, int_t lf1);
};

}
}

#endif
