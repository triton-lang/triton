#ifndef ISAAC_BACKEND_TEMPLATES_MAXPY_H
#define ISAAC_BACKEND_TEMPLATES_MAXPY_H

#include <vector>
#include "isaac/kernels/templates/base.h"

namespace isaac
{
namespace templates
{

class ger_parameters : public base::parameters_type
{
public:
  ger_parameters(unsigned int _simd_width, unsigned int _local_size_0, unsigned int _local_size_1, unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetching_policy);

  unsigned int num_groups_0;
  unsigned int num_groups_1;
  fetching_policy_type fetching_policy;
};

class ger : public base_impl<ger, ger_parameters>
{
private:
  int is_invalid_impl(driver::Device const &, math_expression const  &) const;
  std::string generate_impl(std::string const & suffix, math_expression const  & expressions, driver::Device const & device, mapping_type const & mapping) const;
public:
  ger(parameters_type const & parameters, binding_policy_t binding_policy = BIND_INDEPENDENT);
  ger(unsigned int simd, unsigned int ls1, unsigned int ls2,  unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_INDEPENDENT);
  std::vector<int_t> input_sizes(math_expression const  & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const &);
};

}
}

#endif
