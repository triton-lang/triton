#ifndef ATIDLAS_BACKEND_TEMPLATES_MPRODUCT_H
#define ATIDLAS_BACKEND_TEMPLATES_MPRODUCT_H

#include "atidlas/backend/templates/base.h"

namespace atidlas
{

struct mproduct_parameters : public base::parameters_type
{
  mproduct_parameters(unsigned int simd_width
                            , int_t local_size_0, int_t KL, int_t local_size_1
                            , int_t ms, int_t ks, int_t ns
                            , fetching_policy_type A_fetching_policy, fetching_policy_type B_fetching_policy
                            , int_t local_fetch_0, int_t local_fetch_1);

  int_t kL;

  int_t mS;
  int_t kS;
  int_t nS;

  fetching_policy_type A_fetching_policy;
  fetching_policy_type B_fetching_policy;

  int_t local_fetch_0;
  int_t local_fetch_1;

  int_t mL;
  int_t nL;
};

class mproduct : public base_impl<mproduct, mproduct_parameters>
{
private:
  unsigned int lmem_usage(symbolic_expressions_container const & symbolic_expressions) const;
  unsigned int registers_usage(symbolic_expressions_container const & symbolic_expressions) const;
  int check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const;
  std::string generate_impl(unsigned int label, const char *  id, const symbolic_expressions_container &symbolic_expressions, const std::vector<mapping_type> &, bool fallback) const;
  std::vector<std::string> generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const;
  void enqueue_block(cl::CommandQueue & queue, int_t M, int_t N, int_t K,
                     array_infos const & A, array_infos const & B, array_infos const & C,
                     value_scalar const & alpha, value_scalar const & beta,
                     std::vector<cl::lazy_compiler> & programs, unsigned int label, int id);
  array_infos create_slice(array_infos & M, int_t s0_0, int_t s0_1, int_t s1_0, int_t s1_1, bool swap);
  std::vector<int_t> infos(symbolic_expressions_container const & symbolic_expressions,
                                   lhs_rhs_element & C, lhs_rhs_element & A, lhs_rhs_element & B);
public:
  mproduct(mproduct::parameters_type const & parameters, char A_trans, char B_trans);
  std::vector<int_t> input_sizes(symbolic_expressions_container const & symbolic_expressions);
  void enqueue(cl::CommandQueue & queue,
               std::vector<cl::lazy_compiler> & programs,
               unsigned int label,
               symbolic_expressions_container const & symbolic_expressions);

private:
  const char A_trans_;
  const char B_trans_;
};

class mproduct_nn : public mproduct
{
public:
  mproduct_nn(mproduct::parameters_type  const &);
  mproduct_nn(unsigned int simd, int_t ls0, int_t KL, int_t ls1
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1);
};

class mproduct_tn : public mproduct
{
public:
  mproduct_tn(mproduct::parameters_type  const &);
  mproduct_tn(unsigned int simd, int_t ls0, int_t KL, int_t ls1
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1);
};


class mproduct_nt : public mproduct
{
public:
  mproduct_nt(mproduct::parameters_type  const &);
  mproduct_nt(unsigned int simd, int_t ls0, int_t KL, int_t ls1
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1);
};


class mproduct_tt : public mproduct
{
public:
  mproduct_tt(mproduct::parameters_type  const &);
  mproduct_tt(unsigned int simd, int_t ls0, int_t KL, int_t ls1
                      , int_t ms, int_t ks, int_t ns, fetching_policy_type Afetch , fetching_policy_type Bfetch
                      , int_t lfetch0, int_t lfetch1);
};


}

#endif
