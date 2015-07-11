#ifndef ISAAC_BACKEND_TEMPLATES_VAXPY_H
#define ISAAC_BACKEND_TEMPLATES_VAXPY_H

#include "isaac/backend/templates/base.h"

namespace isaac
{
namespace templates
{

class axpy_parameters : public base::parameters_type
{
public:
  axpy_parameters(unsigned int _simd_width, unsigned int _group_size, unsigned int _num_groups, fetching_policy_type _fetching_policy);
  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class axpy : public base_impl<axpy, axpy_parameters>
{
private:
  virtual int is_invalid_impl(driver::Device const &, expressions_tuple const &) const;
  std::string generate_impl(const char * suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const & mappings) const;
public:
  axpy(axpy::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  axpy(unsigned int _simd_width, unsigned int _group_size, unsigned int _num_groups, fetching_policy_type _fetching_policy, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  std::vector<int_t> input_sizes(expressions_tuple const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program & program, const char * suffix, base & fallback, controller<expressions_tuple> const &);
};

}
}

#endif
