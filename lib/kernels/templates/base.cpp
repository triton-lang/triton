#include <cassert>
#include <algorithm>

#include "isaac/array.h"
#include "isaac/kernels/keywords.h"
#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/templates/dot.h"
#include "isaac/kernels/templates/ger.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/kernels/templates/gemm.h"
#include "isaac/kernels/templates/base.h"
#include "isaac/kernels/parse.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/exception/unknown_datatype.h"
#include "isaac/symbolic/io.h"

#include "tools/map.hpp"

#include "to_string.hpp"

namespace isaac
{
namespace templates
{

base::parameters_type::parameters_type(unsigned int _simd_width, int_t _local_size_1, int_t _local_size_2, int_t _num_kernels) : simd_width(_simd_width), local_size_0(_local_size_1), local_size_1(_local_size_2), num_kernels(_num_kernels)
{ }


bool base::requires_fallback(expressions_tuple const & expressions)
{
  for (const auto & elem : expressions.data())
    for(array_expression::container_type::const_iterator itt = (elem)->tree().begin(); itt != (elem)->tree().end() ; ++itt)
      if(   (itt->lhs.subtype==DENSE_ARRAY_TYPE && (std::max(itt->lhs.array->stride()[0], itt->lhs.array->stride()[1])>1 || std::max(itt->lhs.array->start()[0],itt->lhs.array->start()[1])>0))
         || (itt->rhs.subtype==DENSE_ARRAY_TYPE && (std::max(itt->rhs.array->stride()[0], itt->rhs.array->stride()[1])>1 || std::max(itt->rhs.array->start()[0],itt->rhs.array->start()[1])>0)))
        return true;
  return false;
}

int_t base::vector_size(array_expression::node const & node)
{
  using namespace tools;
  if (node.op.type==OPERATOR_MATRIX_DIAG_TYPE)
    return std::min<int_t>(node.lhs.array->shape()[0], node.lhs.array->shape()[1]);
  else if (node.op.type==OPERATOR_MATRIX_ROW_TYPE)
    return node.lhs.array->shape()[1];
  else if (node.op.type==OPERATOR_MATRIX_COLUMN_TYPE)
    return node.lhs.array->shape()[0];
  else
    return std::max(node.lhs.array->shape()[0], node.lhs.array->shape()[1]);

}

std::pair<int_t, int_t> base::matrix_size(array_expression::node const & node)
{
  if (node.op.type==OPERATOR_VDIAG_TYPE)
  {
    int_t size = node.lhs.array->shape()[0];
    return std::make_pair(size,size);
  }
  else if(node.op.type==OPERATOR_REPEAT_TYPE)
    return std::make_pair(node.lhs.array->shape()[0]*node.rhs.tuple.rep1, node.lhs.array->shape()[1]*node.rhs.tuple.rep2);
  else
    return std::make_pair(node.lhs.array->shape()[0],node.lhs.array->shape()[1]);
}


base::base(binding_policy_t binding_policy) : binding_policy_(binding_policy)
{}

unsigned int base::lmem_usage(expressions_tuple const &) const
{ return 0; }

unsigned int base::registers_usage(expressions_tuple const &) const
{ return 0; }

base::~base()
{ }

std::string base::generate(std::string const & suffix, expressions_tuple const & expressions, driver::Device const & device)
{
  expressions_tuple::data_type::const_iterator sit;
  std::vector<mapping_type>::iterator mit;

  if(int err = is_invalid(expressions, device))
    throw operation_not_supported_exception("The supplied parameters for this template are invalid : err " + tools::to_string(err));

  //Create mapping
  std::vector<mapping_type> mappings(expressions.data().size());
  std::unique_ptr<symbolic_binder> binder;
  if (binding_policy_==BIND_TO_HANDLE)
      binder.reset(new bind_to_handle());
  else
      binder.reset(new bind_all_unique());

  for (mit = mappings.begin(), sit = expressions.data().begin(); sit != expressions.data().end(); ++sit, ++mit)
    traverse(**sit, (*sit)->root(), map_functor(*binder,*mit,device), true);

  return generate_impl(suffix, expressions, device, mappings);
}

template<class TType, class PType>
int base_impl<TType, PType>::is_invalid_impl(driver::Device const &, expressions_tuple const &) const
{ return TEMPLATE_VALID; }

template<class TType, class PType>
base_impl<TType, PType>::base_impl(parameters_type const & parameters, binding_policy_t binding_policy) : base(binding_policy), p_(parameters)
{ }

template<class TType, class PType>
int_t base_impl<TType, PType>::local_size_0() const
{ return p_.local_size_0; }

template<class TType, class PType>
int_t base_impl<TType, PType>::local_size_1() const
{ return p_.local_size_1; }

template<class TType, class PType>
std::shared_ptr<base> base_impl<TType, PType>::clone() const
{ return std::shared_ptr<base>(new TType(*dynamic_cast<TType const *>(this))); }

template<class TType, class PType>
int base_impl<TType, PType>::is_invalid(expressions_tuple const & expressions, driver::Device const & device) const
{
  //Query device informations
  size_t lmem_available = device.local_mem_size();
  size_t lmem_used = lmem_usage(expressions);
  if (lmem_used>lmem_available)
    return TEMPLATE_LOCAL_MEMORY_OVERFLOW;

  //Invalid work group size
  size_t max_workgroup_size = device.max_work_group_size();
  std::vector<size_t> max_work_item_sizes = device.max_work_item_sizes();
  if (p_.local_size_0*p_.local_size_1 > max_workgroup_size)
    return TEMPLATE_WORK_GROUP_SIZE_OVERFLOW;
  if (p_.local_size_0 > max_work_item_sizes[0])
    return TEMPLATE_LOCAL_SIZE_0_OVERFLOW;

  if (p_.local_size_1 > max_work_item_sizes[1])
    return TEMPLATE_LOCAL_SIZE_1_OVERFLOW;

  //Invalid SIMD Width
  if (p_.simd_width!=1 && p_.simd_width!=2 && p_.simd_width!=3 && p_.simd_width!=4)
    return TEMPLATE_INVALID_SIMD_WIDTH;

  return is_invalid_impl(device, expressions);
}

template class base_impl<axpy, axpy_parameters>;
template class base_impl<dot, dot_parameters>;
template class base_impl<ger, ger_parameters>;
template class base_impl<gemv, gemv_parameters>;
template class base_impl<gemm, gemm_parameters>;

}
}
