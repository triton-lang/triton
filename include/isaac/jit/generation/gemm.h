/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
