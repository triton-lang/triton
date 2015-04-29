#ifndef ISAAC_BACKEND_TEMPLATES_REDUCTION_H
#define ISAAC_BACKEND_TEMPLATES_REDUCTION_H

#include "isaac/backend/templates/base.h"

namespace isaac
{

struct reduction_parameters : public base::parameters_type
{
  reduction_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy);
  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class reduction : public base_impl<reduction, reduction_parameters>
{
private:
  unsigned int lmem_usage(expressions_tuple const & expressions) const;
  int is_invalid_impl(driver::Device const &, expressions_tuple const &) const;
  inline void reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<mapped_scalar_reduction*> exprs,
                                     std::string const & buf_str, std::string const & buf_value_str, driver::backend_type backend) const;
  std::string generate_impl(const char * suffix,  expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const & mappings) const;

public:
  reduction(reduction::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  reduction(unsigned int simd, unsigned int ls, unsigned int ng, fetching_policy_type fetch, binding_policy_t bind = BIND_ALL_UNIQUE);
  std::vector<int_t> input_sizes(expressions_tuple const & expressions);
  void enqueue(driver::CommandQueue & queue, driver::Program & program, const char * suffix, base & fallback, controller<expressions_tuple> const &);
private:
  std::vector< driver::Buffer > tmp_;
  std::vector< driver::Buffer > tmpidx_;
};

}

#endif
