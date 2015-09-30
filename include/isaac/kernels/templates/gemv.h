#ifndef ISAAC_BACKEND_TEMPLATES_MDOT_H
#define ISAAC_BACKEND_TEMPLATES_MDOT_H

#include <vector>

#include "isaac/symbolic/expression.h"
#include "isaac/kernels/templates/base.h"

namespace isaac
{
namespace templates
{
struct gemv_parameters : public base::parameters_type
{
  gemv_parameters(unsigned int _simd_width,
                                unsigned int _local_size_0, unsigned int _local_size_1,
                                unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetch_policy);
  unsigned int num_groups_0;
  unsigned int num_groups_1;
  fetching_policy_type fetch_policy;
};


class gemv : public base_impl<gemv, gemv_parameters>
{
protected:
  enum dot_type
  {
    REDUCE_ROWS,
    REDUCE_COLUMNS
  };
  gemv(gemv::parameters_type const & , dot_type, binding_policy_t);
private:
  virtual int is_invalid_impl(driver::Device const &, math_expression const &) const;
  unsigned int lmem_usage(math_expression const &) const;
  std::string generate_impl(std::string const & suffix, math_expression const &, driver::Device const & device, mapping_type const &) const;
public:
  virtual std::vector<int_t> input_sizes(math_expression const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const &);
private:
  dot_type dot_type_;
};

class gemv_n : public gemv
{
public:
  gemv_n(gemv::parameters_type  const &, binding_policy_t binding_policy = BIND_INDEPENDENT);
  gemv_n(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_INDEPENDENT);
};

class gemv_t : public gemv
{
public:
  gemv_t(gemv::parameters_type  const &, binding_policy_t binding_policy = BIND_INDEPENDENT);
  gemv_t(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_INDEPENDENT);
};

}
}

#endif
