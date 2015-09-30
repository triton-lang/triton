#ifndef ISAAC_TEMPLATES_base_
#define ISAAC_TEMPLATES_base_


#include <list>
#include <set>
#include <cmath>

#include "isaac/types.h"
#include "isaac/kernels/parse.h"
#include "isaac/kernels/stream.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

namespace templates
{

enum fetching_policy_type
{
  FETCH_FROM_LOCAL,
  FETCH_FROM_GLOBAL_STRIDED,
  FETCH_FROM_GLOBAL_CONTIGUOUS
};

//Error codes
static const int TEMPLATE_VALID = 0;
static const int TEMPLATE_LOCAL_MEMORY_OVERFLOW = -1;
static const int TEMPLATE_WORK_GROUP_SIZE_OVERFLOW = -2;
static const int TEMPLATE_LOCAL_SIZE_0_OVERFLOW = -3;
static const int TEMPLATE_LOCAL_SIZE_1_OVERFLOW = -4;
static const int TEMPLATE_LOCAL_SIZE_2_OVERFLOW = -5;
static const int TEMPLATE_LOCAL_SIZE_NOT_WARP_MULTIPLE = -6;
static const int TEMPLATE_INVALID_SIMD_WIDTH = -7;
static const int TEMPLATE_ALIGNMENT_MUST_BE_BLOCK_SIZE_MULTIPLE = -8;
static const int TEMPLATE_INVALID_FETCHING_POLICY_TYPE= -9;

static const int TEMPLATE_GLOBAL_MEMORY_REQUIRES_ZERO_LOCAL_FETCH = -10;
static const int TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE = -11;
static const int TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL = -12;
static const int TEMPLATE_SIMD_WIDTH_MUST_BE_ONE = -13;
static const int TEMPLATE_LOCAL_FETCH_PRODUCT_MUST_MATCH_LOCAL_SIZE_PRODUCT = -14;
static const int TEMPLATE_LOCAL_FETCH_0_MUST_BE_KL_MULTIPLE = -15;
static const int TEMPLATE_LOCAL_FETCH_0_MUST_BE_NL_MULTIPLE = -16;
static const int TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE = -17;
static const int TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE = -18;
static const int TEMPLATE_TEMPORARY_TOO_LARGE = -19;
static const int TEMPLATE_BLOCK_SIZE_TOO_LARGE = -20;

class base
{
public:
  struct parameters_type
  {
    parameters_type(unsigned int _simd_width, int_t _local_size_1, int_t _local_size_2, int_t _num_kernels);
    unsigned int simd_width;
    unsigned int local_size_0;
    unsigned int local_size_1;
    unsigned int num_kernels;
  };
protected:
  static int_t vector_size(math_expression::node const & node);
  static std::pair<int_t, int_t> matrix_size(math_expression::container_type const & tree, math_expression::node const & node);
  static bool requires_fallback(math_expression const & expressions);
private:
  virtual std::string generate_impl(std::string const & suffix, math_expression const & expressions, driver::Device const & device, mapping_type const & mapping) const = 0;
public:
  base(binding_policy_t binding_policy);
  virtual unsigned int temporary_workspace(math_expression const &) const;
  virtual unsigned int lmem_usage(math_expression const &) const;
  virtual unsigned int registers_usage(math_expression const &) const;
  virtual std::vector<int_t> input_sizes(math_expression const & expressions) const = 0;
  virtual ~base();
  std::string generate(std::string const & suffix, math_expression const & expressions, driver::Device const & device);
  virtual int is_invalid(math_expression const & expressions, driver::Device const & device) const = 0;
  virtual void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const & expressions) = 0;
  virtual std::shared_ptr<base> clone() const = 0;
private:
  binding_policy_t binding_policy_;
};


template<class TemplateType, class ParametersType>
class base_impl : public base
{
private:
  virtual int is_invalid_impl(driver::Device const &, math_expression const &) const;
public:
  typedef ParametersType parameters_type;
  base_impl(parameters_type const & parameters, binding_policy_t binding_policy);
  unsigned int local_size_0() const;
  unsigned int local_size_1() const;
  std::shared_ptr<base> clone() const;
  /** @brief returns whether or not the profile has undefined behavior on particular device */
  int is_invalid(math_expression const & expressions, driver::Device const & device) const;
protected:
  parameters_type p_;
  binding_policy_t binding_policy_;
};

}
}

#endif
