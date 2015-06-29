#ifndef ISAAC_BACKEND_TEMPLATES_MREDUCTION_H
#define ISAAC_BACKEND_TEMPLATES_MREDUCTION_H

#include <vector>

#include "isaac/symbolic/expression.h"
#include "isaac/backend/templates/base.h"

namespace isaac
{

struct mreduction_parameters : public base::parameters_type
{
  mreduction_parameters(unsigned int _simd_width,
                                unsigned int _local_size_0, unsigned int _local_size_1,
                                unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetch_policy);
  unsigned int num_groups_0;
  unsigned int num_groups_1;
  fetching_policy_type fetch_policy;
};


class mreduction : public base_impl<mreduction, mreduction_parameters>
{
protected:
  enum reduction_type
  {
    REDUCE_ROWS,
    REDUCE_COLUMNS
  };
  mreduction(mreduction::parameters_type const & , reduction_type, binding_policy_t);
private:
  virtual int is_invalid_impl(driver::Device const &, expressions_tuple const &) const;
  unsigned int lmem_usage() const;
  std::string generate_impl(const char * suffix, expressions_tuple const &, driver::Device const & device, std::vector<mapping_type> const &) const;
public:
  virtual std::vector<int_t> input_sizes(expressions_tuple const & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program & program, const char * suffix, base & fallback, controller<expressions_tuple> const &);
private:
  reduction_type reduction_type_;
};

class mreduction_rows : public mreduction
{
public:
  mreduction_rows(mreduction::parameters_type  const &, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  mreduction_rows(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_ALL_UNIQUE);
};

class mreduction_cols : public mreduction
{
public:
  mreduction_cols(mreduction::parameters_type  const &, binding_policy_t binding_policy = BIND_ALL_UNIQUE);
  mreduction_cols(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind = BIND_ALL_UNIQUE);
};

}

#endif
