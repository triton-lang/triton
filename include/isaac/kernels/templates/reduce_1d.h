#ifndef ISAAC_BACKEND_TEMPLATES_DOT_H
#define ISAAC_BACKEND_TEMPLATES_DOT_H

#include "isaac/kernels/templates/base.h"

namespace isaac
{
namespace templates
{

struct reduce_1d_parameters : public base::parameters_type
{
  reduce_1d_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy);
  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class reduce_1d : public base_impl<reduce_1d, reduce_1d_parameters>
{
private:
  unsigned int lmem_usage(math_expression const  & expressions) const;
  int is_invalid_impl(driver::Device const &, math_expression const  &) const;
  inline void reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<mapped_reduce_1d*> exprs,
                                     std::string const & buf_str, std::string const & buf_value_str, driver::backend_type backend) const;
  std::string generate_impl(std::string const & suffix,  math_expression const  & expressions, driver::Device const & device, mapping_type const & mapping) const;

public:
  reduce_1d(reduce_1d::parameters_type const & parameters, binding_policy_t binding_policy = BIND_INDEPENDENT);
  reduce_1d(unsigned int simd, unsigned int ls, unsigned int ng, fetching_policy_type fetch, binding_policy_t bind = BIND_INDEPENDENT);
  std::vector<int_t> input_sizes(math_expression const  & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const &);
private:
  std::vector< driver::Buffer > tmp_;
  std::vector< driver::Buffer > tmpidx_;
};

}
}

#endif
