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

class cublas_gemm : public external_base
{
  bool init();
public:
  cublas_gemm(char A_trans, char B_trans);
  int is_invalid(expression_tree const  &, driver::Device const &) const;
  std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const &, std::string const &, runtime::execution_handler const & h);
  expression_type type() const;
private:
  const char A_trans_;
  const char B_trans_;
  bool init_;
};

class intelblas_gemm : public external_base
{
  bool init();
public:
  intelblas_gemm(char A_trans, char B_trans);
  int is_invalid(expression_tree const  &, driver::Device const &) const;
  std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const &, std::string const &, runtime::execution_handler const & h);
  expression_type type() const;
private:
  std::string generate_impl(std::string const & suffix, expression_tree const &, driver::Device const & device, symbolic::symbols_table const &) const;
  const char A_trans_;
  const char B_trans_;
  bool init_;
};

class intelblas_gemm_image : public external_base
{
  bool init();
public:
  intelblas_gemm_image(char A_trans, char B_trans);
  int is_invalid(expression_tree const  &, driver::Device const &) const;
  std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const &, std::string const &, runtime::execution_handler const & h);
  expression_type type() const;
private:
  std::string generate_impl(std::string const & suffix, expression_tree const &, driver::Device const & device, symbolic::symbols_table const &) const;
  const char A_trans_;
  const char B_trans_;
  bool init_;
};


class gemm : public parameterized_base
{
private:
  unsigned int temporary_workspace(expression_tree const & expressions) const;
  unsigned int lmem_usage(expression_tree const & expressions) const;
  unsigned int registers_usage(expression_tree const & expressions) const;
  int is_invalid_impl(driver::Device const &, expression_tree const &) const;
  std::string generate_impl(std::string const & suffix, expression_tree const & expressions, driver::Device const & device, symbolic::symbols_table const &) const;
  void enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K, const expression_tree::node &A, const expression_tree::node &B, const expression_tree::node &C,
                     value_scalar const &alpha, value_scalar const &beta, driver::Program const & program, std::string const & suffix, runtime::execution_options_type const & options);

public:
  gemm(unsigned int simd, int_t ls0, int_t KL, int_t ls1, int_t D
       , int_t ms, int_t ks, int_t ns, int_t lf0, int_t lf1
       , char A_trans, char B_trans);
  std::vector<int_t> input_sizes(expression_tree const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & h);
  expression_type type() const;

private:
  //Parameters
  unsigned int mL_;
  unsigned int kL_;
  unsigned int nL_;
  unsigned int depth_;
  unsigned int mS_;
  unsigned int kS_;
  unsigned int nS_;

  unsigned int lf0_;
  unsigned int lf1_;

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
  gemm_nn(unsigned int vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, int_t lf0, int_t lf1);
};

class gemm_tn : public gemm
{
public:
  gemm_tn(unsigned int vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, int_t lf0, int_t lf1);
};


class gemm_nt : public gemm
{
public:
  gemm_nt(unsigned int vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, int_t lf0, int_t lf1);
};


class gemm_tt : public gemm
{
public:
  gemm_tt(unsigned int vwidth, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, int_t lf0, int_t lf1);
};

}
}

#endif
