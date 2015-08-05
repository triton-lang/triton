#ifndef ISAAC_BACKEND_TEMPLATES_MPRODUCT_H
#define ISAAC_BACKEND_TEMPLATES_MPRODUCT_H

#include "isaac/kernels/templates/base.h"
#include "isaac/symbolic/expression.h"
#include "isaac/symbolic/preset.h"

namespace isaac
{
namespace templates
{
class model;

struct gemm_parameters : public base::parameters_type
{
  gemm_parameters(unsigned int simd_width
                            , unsigned int local_size_0, unsigned int KL, unsigned int local_size_1, unsigned int D
                            , unsigned int ms, unsigned int ks, unsigned int ns
                            , fetching_policy_type A_fetching_policy, fetching_policy_type B_fetching_policy
                            , unsigned int local_fetch_0, unsigned int local_fetch_1);

  unsigned int kL;
  unsigned int depth;

  unsigned int mS;
  unsigned int kS;
  unsigned int nS;

  fetching_policy_type A_fetching_policy;
  fetching_policy_type B_fetching_policy;

  unsigned int local_fetch_0;
  unsigned int local_fetch_1;

  unsigned int mL;
  unsigned int nL;

  bool prefetch;
  bool unroll_outer;
};

class gemm : public base_impl<gemm, gemm_parameters>
{
private:
  unsigned int lmem_usage(expressions_tuple const & expressions) const;
  unsigned int registers_usage(expressions_tuple const & expressions) const;
  int is_invalid_impl(driver::Device const &, expressions_tuple const &) const;
  std::string generate_impl(std::string const & suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const &) const;
  void enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K, array const & A, array const & B, array const & C,
                     value_scalar const &alpha, value_scalar const &beta, driver::Program const & program, std::string const & suffix, execution_options_type const & options);
  array create_slice(array & M, int_t s0_0, int_t s0_1, int_t s1_0, int_t s1_1, bool swap);
  std::vector<int_t> infos(expressions_tuple const & expressions,  isaac::symbolic::preset::gemm::args &arguments) const;
public:
  gemm(gemm::parameters_type const & parameters, bool check_bound, char A_trans, char B_trans);
  std::vector<int_t> input_sizes(expressions_tuple const & expressions) const;
  void cleanup(values_holder beta, controller<expressions_tuple> const & ctr, model & fallback,
               lhs_rhs_element* eA, lhs_rhs_element* eB, lhs_rhs_element* eC, lhs_rhs_element* ebeta, array const & A, array const & B, array const & C);
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, controller<expressions_tuple> const &ctr);
private:
  const char A_trans_;
  const char B_trans_;
  expression_type type_;
  bool check_bounds_;
};

class gemm_nn : public gemm
{
public:
  gemm_nn(unsigned int simd, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1, bool check_bound = false);
};

class gemm_tn : public gemm
{
public:
  gemm_tn(unsigned int simd, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1, bool check_bound = false);
};


class gemm_nt : public gemm
{
public:
  gemm_nt(unsigned int simd, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1, bool check_bound = false);
};


class gemm_tt : public gemm
{
public:
  gemm_tt(unsigned int simd, int_t ls0, int_t KL, int_t ls1, int_t D
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1, bool check_bound = false);
};

}
}

#endif
