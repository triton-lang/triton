#ifndef ATIDLAS_BACKEND_TEMPLATES_VAXPY_H
#define ATIDLAS_BACKEND_TEMPLATES_VAXPY_H

#include "atidlas/backend/templates/base.h"

namespace atidlas
{

class vaxpy_parameters : public base::parameters_type
{
public:
  vaxpy_parameters(unsigned int _simd_width, unsigned int _group_size, unsigned int _num_groups, fetching_policy_type _fetching_policy);
  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class vaxpy : public base_impl<vaxpy, vaxpy_parameters>
{
private:
  virtual int check_invalid_impl(cl::Device const &, expressions_tuple const &) const;
  std::vector<std::string> generate_impl(unsigned int label, expressions_tuple const & expressions, std::vector<mapping_type> const & mappings) const;
public:
  vaxpy(vaxpy::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  vaxpy(unsigned int _simd_width, unsigned int _group_size, unsigned int _num_groups, fetching_policy_type _fetching_policy, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  std::vector<int_t> input_sizes(expressions_tuple const & expressions);
  void enqueue(cl::CommandQueue & queue, std::vector<cl_ext::lazy_compiler> & programs, unsigned int label, expressions_tuple const & expressions, operation_cache * cache = NULL);
};

}

#endif
