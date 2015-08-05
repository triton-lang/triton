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
#include "isaac/tools/to_string.hpp"
#include "isaac/tools/make_map.hpp"
#include "isaac/symbolic/io.h"

namespace isaac
{
namespace templates
{

base::parameters_type::parameters_type(unsigned int _simd_width, int_t _local_size_1, int_t _local_size_2, int_t _num_kernels) : simd_width(_simd_width), local_size_0(_local_size_1), local_size_1(_local_size_2), num_kernels(_num_kernels)
{ }

numeric_type base::map_functor::get_numeric_type(isaac::array_expression const * array_expression, int_t root_idx) const
{
  array_expression::node const * root_node = &array_expression->tree()[root_idx];
  while (root_node->lhs.dtype==INVALID_NUMERIC_TYPE)
    root_node = &array_expression->tree()[root_node->lhs.node_index];
  return root_node->lhs.dtype;
}

/** @brief Binary leaf */
template<class T>
std::shared_ptr<mapped_object> base::map_functor::binary_leaf(isaac::array_expression const * array_expression, int_t root_idx, mapping_type const * mapping) const
{
  return std::shared_ptr<mapped_object>(new T(numeric_type_to_string(array_expression->dtype()), binder_.get(), mapped_object::node_info(mapping, array_expression, root_idx)));
}

/** @brief Scalar mapping */
std::shared_ptr<mapped_object> base::map_functor::create(numeric_type dtype, values_holder) const
{
  std::string strdtype = numeric_type_to_string(dtype);
  return std::shared_ptr<mapped_object>(new mapped_host_scalar(strdtype, binder_.get()));
}

/** @brief Vector mapping */
std::shared_ptr<mapped_object> base::map_functor::create(array const * a) const
{
  std::string dtype = numeric_type_to_string(a->dtype());
  unsigned int id = binder_.get(a->data());
  //Scalar
  if(a->shape()[0]==1 && a->shape()[1]==1)
    return std::shared_ptr<mapped_object>(new mapped_array(dtype, id, 's'));
  //Column vector
  else if(a->shape()[0]>1 && a->shape()[1]==1)
    return std::shared_ptr<mapped_object>(new mapped_array(dtype, id, 'c'));
  //Row vector
  else if(a->shape()[0]==1 && a->shape()[1]>1)
    return std::shared_ptr<mapped_object>(new mapped_array(dtype, id, 'r'));
  //Matrix
  else
    return std::shared_ptr<mapped_object>(new mapped_array(dtype, id, 'm'));
}

std::shared_ptr<mapped_object> base::map_functor::create(repeat_infos const &) const
{
  //TODO: Make it less specific!
  return std::shared_ptr<mapped_object>(new mapped_tuple(size_type(device_),binder_.get(),4));
}

std::shared_ptr<mapped_object> base::map_functor::create(lhs_rhs_element const & lhs_rhs) const
{
  switch(lhs_rhs.type_family)
  {
    case INFOS_TYPE_FAMILY: return create(lhs_rhs.tuple);
    case VALUE_TYPE_FAMILY: return create(lhs_rhs.dtype, lhs_rhs.vscalar);
    case ARRAY_TYPE_FAMILY: return create(lhs_rhs.array);
    default: throw "";
  }
}


base::map_functor::map_functor(symbolic_binder & binder, mapping_type & mapping, driver::Device const & device) : binder_(binder), mapping_(mapping), device_(device){ }

/** @brief Traversal functor */
void base::map_functor::operator()(isaac::array_expression const & array_expression, int_t root_idx, leaf_t leaf_t) const
{
  mapping_type::key_type key(root_idx, leaf_t);
  array_expression::node const & root_node = array_expression.tree()[root_idx];

  if (leaf_t == LHS_NODE_TYPE && root_node.lhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    mapping_.insert(mapping_type::value_type(key, create(root_node.lhs)));
  else if (leaf_t == RHS_NODE_TYPE && root_node.rhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    mapping_.insert(mapping_type::value_type(key, create(root_node.rhs)));
  else if ( leaf_t== PARENT_NODE_TYPE)
  {
    if (root_node.op.type==OPERATOR_VDIAG_TYPE)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_vdiag>(&array_expression, root_idx, &mapping_)));
    else if (root_node.op.type==OPERATOR_MATRIX_DIAG_TYPE)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_diag>(&array_expression, root_idx, &mapping_)));
    else if (root_node.op.type==OPERATOR_MATRIX_ROW_TYPE)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_row>(&array_expression, root_idx, &mapping_)));
    else if (root_node.op.type==OPERATOR_MATRIX_COLUMN_TYPE)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_column>(&array_expression, root_idx, &mapping_)));
    else if (detail::is_scalar_dot(root_node))
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_scalar_dot>(&array_expression, root_idx, &mapping_)));
    else if (detail::is_vector_dot(root_node))
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_gemv>(&array_expression, root_idx, &mapping_)));
    else if (root_node.op.type_family == OPERATOR_GEMM_TYPE_FAMILY)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_gemm>(&array_expression, root_idx, &mapping_)));
    else if (root_node.op.type == OPERATOR_REPEAT_TYPE)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_repeat>(&array_expression, root_idx, &mapping_)));
    else if (root_node.op.type == OPERATOR_OUTER_PROD_TYPE)
      mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_outer>(&array_expression, root_idx, &mapping_)));
    else if (detail::is_cast(root_node.op))
      mapping_.insert(mapping_type::value_type(key, std::shared_ptr<mapped_object>(new mapped_cast(root_node.op.type, binder_.get()))));
  }
}

base::set_arguments_functor::set_arguments_functor(symbolic_binder & binder, unsigned int & current_arg, driver::Kernel & kernel) :
  binder_(binder), current_arg_(current_arg), kernel_(kernel){ }

void base::set_arguments_functor::set_arguments(numeric_type dtype, values_holder const & scal) const
{
  switch(dtype)
  {
//    case BOOL_TYPE: kernel_.setArg(current_arg_++, scal.bool8); break;
    case CHAR_TYPE: kernel_.setArg(current_arg_++, scal.int8); break;
    case UCHAR_TYPE: kernel_.setArg(current_arg_++, scal.uint8); break;
    case SHORT_TYPE: kernel_.setArg(current_arg_++, scal.int16); break;
    case USHORT_TYPE: kernel_.setArg(current_arg_++, scal.uint16); break;
    case INT_TYPE: kernel_.setArg(current_arg_++, scal.int32); break;
    case UINT_TYPE: kernel_.setArg(current_arg_++, scal.uint32); break;
    case LONG_TYPE: kernel_.setArg(current_arg_++, scal.int64); break;
    case ULONG_TYPE: kernel_.setArg(current_arg_++, scal.uint64); break;
//    case HALF_TYPE: kernel_.setArg(current_arg_++, scal.float16); break;
    case FLOAT_TYPE: kernel_.setArg(current_arg_++, scal.float32); break;
    case DOUBLE_TYPE: kernel_.setArg(current_arg_++, scal.float64); break;
    default: throw unknown_datatype(dtype);
  }
}

/** @brief Vector mapping */
void base::set_arguments_functor::set_arguments(array const * a) const
{
  bool is_bound = binder_.bind(a->data());
  if (is_bound)
  {
    kernel_.setArg(current_arg_++, a->data());
    //scalar
    if(a->shape()[0]==1 && a->shape()[1]==1)
    {
      kernel_.setSizeArg(current_arg_++, a->start()[0]);
    }
    //array
    else if(a->shape()[0]==1 || a->shape()[1]==1)
    {
      kernel_.setSizeArg(current_arg_++, std::max(a->start()[0], a->start()[1]));
      kernel_.setSizeArg(current_arg_++, std::max(a->stride()[0], a->stride()[1]));
    }
    else
    {
      kernel_.setSizeArg(current_arg_++, a->ld());
      kernel_.setSizeArg(current_arg_++, a->start()[0]);
      kernel_.setSizeArg(current_arg_++, a->start()[1]);
      kernel_.setSizeArg(current_arg_++, a->stride()[0]);
      kernel_.setSizeArg(current_arg_++, a->stride()[1]);
    }
  }
}

void base::set_arguments_functor::set_arguments(repeat_infos const & i) const
{
  kernel_.setSizeArg(current_arg_++, i.sub1);
  kernel_.setSizeArg(current_arg_++, i.sub2);
  kernel_.setSizeArg(current_arg_++, i.rep1);
  kernel_.setSizeArg(current_arg_++, i.rep2);
}

void base::set_arguments_functor::set_arguments(lhs_rhs_element const & lhs_rhs) const
{
  switch(lhs_rhs.type_family)
  {
    case VALUE_TYPE_FAMILY: return set_arguments(lhs_rhs.dtype, lhs_rhs.vscalar);
    case ARRAY_TYPE_FAMILY: return set_arguments(lhs_rhs.array);
    case INFOS_TYPE_FAMILY: return set_arguments(lhs_rhs.tuple);
    default: throw invalid_exception("Unrecognized type family");
  }
}

/** @brief Traversal functor: */
void base::set_arguments_functor::operator()(isaac::array_expression const & array_expression, int_t root_idx, leaf_t leaf_t) const
{
  array_expression::node const & root_node = array_expression.tree()[root_idx];
  if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    set_arguments(root_node.lhs);
  else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    set_arguments(root_node.rhs);
}

void base::compute_dot(kernel_generation_stream & os, std::string acc, std::string cur, op_element const & op)
{
  if (detail::is_elementwise_function(op))
    os << acc << "=" << evaluate(op.type) << "(" << acc << "," << cur << ");" << std::endl;
  else
    os << acc << "= (" << acc << ")" << evaluate(op.type)  << "(" << cur << ");" << std::endl;
}

void base::compute_index_dot(kernel_generation_stream & os, std::string acc, std::string cur, std::string const & acc_value, std::string const & cur_value, op_element const & op)
{
  //        os << acc << " = " << cur_value << ">" << acc_value  << "?" << cur << ":" << acc << ";" << std::endl;
  os << acc << "= select(" << acc << "," << cur << "," << cur_value << ">" << acc_value << ");" << std::endl;
  os << acc_value << "=";
  if (op.type==OPERATOR_ELEMENT_ARGFMAX_TYPE) os << "fmax";
  if (op.type==OPERATOR_ELEMENT_ARGMAX_TYPE) os << "max";
  if (op.type==OPERATOR_ELEMENT_ARGFMIN_TYPE) os << "fmin";
  if (op.type==OPERATOR_ELEMENT_ARGMIN_TYPE) os << "min";
  os << "(" << acc_value << "," << cur_value << ");"<< std::endl;
}

void base::process_all(std::string const & type_key, std::string const & str,
                        kernel_generation_stream & stream, std::vector<mapping_type> const & mappings)
{
  for (const auto & mapping : mappings)
    for (mapping_type::const_iterator mmit = mapping.begin(); mmit != mapping.end(); ++mmit)
      if (mmit->second->type_key()==type_key)
        stream << mmit->second->process(str) << std::endl;
}


void base::base::process_all_at(std::string const & type_key, std::string const & str,
                           kernel_generation_stream & stream, std::vector<mapping_type> const & mappings,
                           size_t root_idx, leaf_t leaf)
{
  for (const auto & mapping : mappings)
  {
    mapped_object * obj = mapping.at(mapping_key(root_idx, leaf)).get();
    if (obj->type_key()==type_key)
      stream << obj->process(str) << std::endl;
  }
}

std::string base::neutral_element(op_element const & op, driver::backend_type backend, std::string const & dtype)
{
  std::string INF = Infinity(backend, dtype).get();
  std::string N_INF = "-" + INF;

  switch (op.type)
  {
  case OPERATOR_ADD_TYPE : return "0";
  case OPERATOR_MULT_TYPE : return "1";
  case OPERATOR_DIV_TYPE : return "1";
  case OPERATOR_ELEMENT_FMAX_TYPE : return N_INF;
  case OPERATOR_ELEMENT_ARGFMAX_TYPE : return N_INF;
  case OPERATOR_ELEMENT_MAX_TYPE : return N_INF;
  case OPERATOR_ELEMENT_ARGMAX_TYPE : return N_INF;
  case OPERATOR_ELEMENT_FMIN_TYPE : return INF;
  case OPERATOR_ELEMENT_ARGFMIN_TYPE : return INF;
  case OPERATOR_ELEMENT_MIN_TYPE : return INF;
  case OPERATOR_ELEMENT_ARGMIN_TYPE : return INF;

  default: throw operation_not_supported_exception("Unsupported dot operator : no neutral element known");
  }
}

std::string base::generate_arguments(std::vector<mapping_type> const & mappings, std::map<std::string, std::string> const & accessors, expressions_tuple const & expressions)
{
  kernel_generation_stream stream;
  process(stream, PARENT_NODE_TYPE, accessors, expressions, mappings);
  std::string res = stream.str();
  res.erase(res.rfind(','));
  return res;
}

std::string base::generate_arguments(std::string const & data_type, driver::Device const & device, std::vector<mapping_type> const & mappings, expressions_tuple const & expressions)
{
  std::string kwglobal = Global(device.backend()).get();
  std::string _size_t = size_type(device);
  return generate_arguments(mappings, tools::make_map<std::map<std::string, std::string> >("array0", kwglobal + " #scalartype* #pointer, " + _size_t + " #start,")
                                                                    ("host_scalar", "#scalartype #name,")
                                                                    ("array1", kwglobal + " " + data_type + "* #pointer, " + _size_t + " #start, " + _size_t + " #stride,")
                                                                    ("array2", kwglobal + " " + data_type + "* #pointer, " + _size_t + " #ld, " + _size_t + " #start1, " + _size_t + " #start2, " + _size_t + " #stride1, " + _size_t + " #stride2,")
                                                                    ("tuple4", "#scalartype #name0, #scalartype #name1, #scalartype #name2, #scalartype #name3,"), expressions);
}



void base::set_arguments(expressions_tuple const & expressions, driver::Kernel & kernel, unsigned int & current_arg)
{
  std::shared_ptr<symbolic_binder> binder = make_binder();
  for (const auto & elem : expressions.data())
    traverse(*elem, (elem)->root(), set_arguments_functor(*binder, current_arg, kernel), true);
}

base::invalid_exception::invalid_exception() : message_() {}

base::invalid_exception::invalid_exception(std::string message) :
  message_("ISAAC: Internal error: The generator cannot apply the given template to the given array_expression: " + message + "\n"
           "If you are using a builtin template, please report on viennacl-support@lists.sourceforge.net! We will provide a fix as soon as possible\n"
           "If you are using your own template, please try using other parameters") {}

const char* base::invalid_exception::what() const throw() { return message_.c_str(); }

base::invalid_exception::~invalid_exception() throw() {}

void base::fetching_loop_info(fetching_policy_type policy, std::string const & bound, kernel_generation_stream & stream, std::string & init, std::string & upper_bound, std::string & inc, std::string const & domain_id, std::string const & domain_size, driver::Device const & device)
{
  if (policy==FETCH_FROM_GLOBAL_STRIDED)
  {
    init = domain_id;
    upper_bound = bound;
    inc = domain_size;
  }
  else if (policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
  {
    std::string _size_t = size_type(device);
    std::string chunk_size = "chunk_size";
    std::string chunk_start = "chunk_start";
    std::string chunk_end = "chunk_end";

    stream << _size_t << " " << chunk_size << " = (" << bound << "+" << domain_size << "-1)/" << domain_size << ";" << std::endl;
    stream << _size_t << " " << chunk_start << " =" << domain_id << "*" << chunk_size << ";" << std::endl;
    stream << _size_t << " " << chunk_end << " = min(" << chunk_start << "+" << chunk_size << ", " << bound << ");" << std::endl;
    init = chunk_start;
    upper_bound = chunk_end;
    inc = "1";
  }
}

bool base::is_node_trans(array_expression::container_type const & array, size_t root_idx, leaf_t leaf_type)
{
  bool res = false;
  lhs_rhs_element array_expression::node::*ptr;
  if (leaf_type==LHS_NODE_TYPE)
    ptr = &array_expression::node::lhs;
  else
    ptr = &array_expression::node::rhs;
  array_expression::node const * node = &array[root_idx];
  while ((node->*ptr).type_family==COMPOSITE_OPERATOR_FAMILY)
  {
    if (array[(node->*ptr).node_index].op.type==OPERATOR_TRANS_TYPE)
      res = !res;
    node = &array[(node->*ptr).node_index];
  }
  return res;
}

std::string base::append_simd_suffix(std::string const & str, unsigned int i)
{
  assert(i < 16);
  char suffixes[] = {'0','1','2','3','4','5','6','7','8','9',
                           'a','b','c','d','e','f'};
  return str + tools::to_string(suffixes[i]);
}

bool base::is_strided(array_expression::node const & node)
{
  return node.op.type==OPERATOR_VDIAG_TYPE
      || node.op.type==OPERATOR_MATRIX_DIAG_TYPE
      || node.op.type==OPERATOR_MATRIX_ROW_TYPE
      || node.op.type==OPERATOR_MATRIX_COLUMN_TYPE
      || node.op.type==OPERATOR_OUTER_PROD_TYPE;
}

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

bool base::is_dot(array_expression::node const & node)
{
  return node.op.type_family==OPERATOR_VECTOR_DOT_TYPE_FAMILY
      || node.op.type_family==OPERATOR_COLUMNS_DOT_TYPE_FAMILY
      || node.op.type_family==OPERATOR_ROWS_DOT_TYPE_FAMILY;
}

bool base::is_index_dot(op_element const & op)
{
  return op.type==OPERATOR_ELEMENT_ARGFMAX_TYPE
      || op.type==OPERATOR_ELEMENT_ARGMAX_TYPE
      || op.type==OPERATOR_ELEMENT_ARGFMIN_TYPE
      || op.type==OPERATOR_ELEMENT_ARGMIN_TYPE;
}

std::string base::access_vector_type(std::string const & v, int i)
{
  switch(i)
  {
    case 0: return v + ".x";
    case 1: return v + ".y";
    case 2: return v + ".z";
    case 3: return v + ".w";
    default: throw;
  }
}

std::string base::vstore(unsigned int simd_width, std::string const &
                         #ifdef ISAAC_WITH_CUDA
                         dtype
                         #endif
                         , std::string const & value, std::string const & offset, std::string const & ptr, driver::backend_type backend)
{
  if (simd_width==1)
    return "(" + ptr + ")[" + offset + "] = " + value;
  else
  {
    switch(backend)
    {
#ifdef ISAAC_WITH_CUDA
      case driver::CUDA:
        return "reinterpret_cast<" + append_width(dtype,simd_width) + "*>(" + ptr + ")[" + offset + "] = " + value;
#endif
      case driver::OPENCL:
        return append_width("vstore", simd_width) + "(" + value + ", " + offset + ", " + ptr + ")";
      default:
        throw;
    }
  }
}

std::string base::vload(unsigned int simd_width, std::string const &
                        #ifdef ISAAC_WITH_CUDA
                        dtype
                        #endif
                        , std::string const & offset, std::string const & ptr, driver::backend_type backend)
{
  if (simd_width==1)
    return "(" + ptr + ")[" + offset + "]";
  else
  {
    switch(backend)
    {
#ifdef ISAAC_WITH_CUDA
      case driver::CUDA:
        return "reinterpret_cast<" + append_width(dtype, simd_width) + "*>(" + ptr + ")[" + offset + "]";
#endif
      case driver::OPENCL:
        return append_width("vload", simd_width) + "(" + offset + ", " + ptr + ")";
      default:
        throw;
    }
  }
}

std::string base::append_width(std::string const & str, unsigned int width)
{
  if (width==1)
    return str;
  return str + tools::to_string(width);
}

std::shared_ptr<symbolic_binder> base::make_binder()
{
  if (binding_policy_==BIND_TO_HANDLE)
    return std::shared_ptr<symbolic_binder>(new bind_to_handle());
  else
    return std::shared_ptr<symbolic_binder>(new bind_all_unique());
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
  std::shared_ptr<symbolic_binder> binder = make_binder();
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
