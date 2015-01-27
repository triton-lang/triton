#ifndef ATIDLAS_BACKEND_TEMPLATES_MAXPY_H
#define ATIDLAS_BACKEND_TEMPLATES_MAXPY_H

#include <vector>
#include "atidlas/backend/templates/base.h"

namespace atidlas
{

class maxpy_parameters : public base::parameters_type
{
public:
  maxpy_parameters(unsigned int _simd_width, unsigned int _local_size_0, unsigned int _local_size_1, unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetching_policy);

  unsigned int num_groups_0;
  unsigned int num_groups_1;
  fetching_policy_type fetching_policy;
};

class maxpy : public base_impl<maxpy, maxpy_parameters>
{
private:
  int check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const;
  std::string generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings, unsigned int simd_width) const;
  std::vector<std::string> generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const;
public:
  maxpy(parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  maxpy(unsigned int simd, unsigned int ls1, unsigned int ls2,  unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_ALL_UNIQUE);
  std::vector<int_t> input_sizes(symbolic_expressions_container const & symbolic_expressions);
  void enqueue(cl::CommandQueue & queue, std::vector<cl::lazy_compiler> & programs,  unsigned int label, symbolic_expressions_container const & symbolic_expressions);
};

}

#endif
