#ifndef ATIDLAS_BACKEND_TEMPLATES_REDUCTION_H
#define ATIDLAS_BACKEND_TEMPLATES_REDUCTION_H

#include "atidlas/backend/templates/template_base.h"

namespace atidlas
{

struct reduction_parameters : public template_base::parameters_type
{
  reduction_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy);
  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class reduction : public template_base_impl<reduction, reduction_parameters>
{
private:
  unsigned int lmem_usage(symbolic_expressions_container const & symbolic_expressions) const;
  int check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const;
  inline void reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<mapped_scalar_reduction*> exprs,
                                     std::string const & buf_str, std::string const & buf_value_str) const;
  std::string generate_impl(unsigned int label, char type, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings, unsigned int simd_width) const;
  std::vector<std::string> generate_impl(unsigned int label,  symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const;

public:
  reduction(reduction::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  reduction(unsigned int simd, unsigned int ls, unsigned int ng, fetching_policy_type fetch, binding_policy_t bind = BIND_ALL_UNIQUE);
  std::vector<int_t> input_sizes(symbolic_expressions_container const & symbolic_expressions);
  void enqueue(cl::CommandQueue & queue,
               std::vector<cl::lazy_compiler> & programs,
               unsigned int label,
               symbolic_expressions_container const & symbolic_expressions);
private:
  std::vector< cl::Buffer > tmp_;
  std::vector< cl::Buffer > tmpidx_;
};

}

#endif
