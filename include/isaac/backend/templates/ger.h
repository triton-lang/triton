#ifndef ISAAC_BACKEND_TEMPLATES_MAXPY_H
#define ISAAC_BACKEND_TEMPLATES_MAXPY_H

#include <vector>
#include "isaac/backend/templates/base.h"

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
  int is_invalid_impl(driver::Device const &, expressions_tuple const &) const;
  std::string generate_impl(const char * suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const & mappings) const;
public:
  ger(parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  ger(unsigned int simd, unsigned int ls1, unsigned int ls2,  unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_ALL_UNIQUE);
  std::vector<int_t> input_sizes(expressions_tuple const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program & program, const char * suffix, base & fallback, controller<expressions_tuple> const &);
};

}
}

#endif
