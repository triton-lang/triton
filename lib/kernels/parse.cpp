#include <cstring>

#include "to_string.hpp"

#include "isaac/array.h"
#include "isaac/kernels/parse.h"
#include "isaac/exception/operation_not_supported.h"

namespace isaac
{

namespace detail
{



  bool is_scalar_dot(math_expression::node const & node)
  {
    return node.op.type_family==OPERATOR_VECTOR_DOT_TYPE_FAMILY;
  }

  bool is_vector_dot(math_expression::node const & node)
  {
    return node.op.type_family==OPERATOR_ROWS_DOT_TYPE_FAMILY
        || node.op.type_family==OPERATOR_COLUMNS_DOT_TYPE_FAMILY;
  }

  bool is_assignment(op_element const & op)
  {
      return op.type== OPERATOR_ASSIGN_TYPE
              || op.type== OPERATOR_INPLACE_ADD_TYPE
              || op.type== OPERATOR_INPLACE_SUB_TYPE;
  }

  bool is_elementwise_operator(op_element const & op)
  {
    return is_assignment(op)
        || op.type== OPERATOR_ADD_TYPE
        || op.type== OPERATOR_SUB_TYPE
        || op.type== OPERATOR_ELEMENT_PROD_TYPE
        || op.type== OPERATOR_ELEMENT_DIV_TYPE
        || op.type== OPERATOR_MULT_TYPE
        || op.type== OPERATOR_DIV_TYPE
        || op.type== OPERATOR_ELEMENT_EQ_TYPE
        || op.type== OPERATOR_ELEMENT_NEQ_TYPE
        || op.type== OPERATOR_ELEMENT_GREATER_TYPE
        || op.type== OPERATOR_ELEMENT_LESS_TYPE
        || op.type== OPERATOR_ELEMENT_GEQ_TYPE
        || op.type== OPERATOR_ELEMENT_LEQ_TYPE ;
  }

  bool bypass(op_element const & op)
  {
        return op.type == OPERATOR_RESHAPE_TYPE
             ||op.type == OPERATOR_TRANS_TYPE;
  }

  bool is_cast(op_element const & op)
  {
        return op.type== OPERATOR_CAST_BOOL_TYPE
            || op.type== OPERATOR_CAST_CHAR_TYPE
            || op.type== OPERATOR_CAST_UCHAR_TYPE
            || op.type== OPERATOR_CAST_SHORT_TYPE
            || op.type== OPERATOR_CAST_USHORT_TYPE
            || op.type== OPERATOR_CAST_INT_TYPE
            || op.type== OPERATOR_CAST_UINT_TYPE
            || op.type== OPERATOR_CAST_LONG_TYPE
            || op.type== OPERATOR_CAST_ULONG_TYPE
            || op.type== OPERATOR_CAST_FLOAT_TYPE
            || op.type== OPERATOR_CAST_DOUBLE_TYPE
            ;
  }

  bool is_node_leaf(op_element const & op)
  {
    return op.type==OPERATOR_MATRIX_DIAG_TYPE
        || op.type==OPERATOR_VDIAG_TYPE
        || op.type==OPERATOR_REPEAT_TYPE
        || op.type==OPERATOR_MATRIX_ROW_TYPE
        || op.type==OPERATOR_MATRIX_COLUMN_TYPE
        || op.type==OPERATOR_ACCESS_INDEX_TYPE
        || op.type==OPERATOR_OUTER_PROD_TYPE
        || op.type_family==OPERATOR_VECTOR_DOT_TYPE_FAMILY
        || op.type_family==OPERATOR_ROWS_DOT_TYPE_FAMILY
        || op.type_family==OPERATOR_COLUMNS_DOT_TYPE_FAMILY
        || op.type_family==OPERATOR_GEMM_TYPE_FAMILY
        ;
  }

  bool is_elementwise_function(op_element const & op)
  {
    return is_cast(op)
        || op.type== OPERATOR_ABS_TYPE
        || op.type== OPERATOR_ACOS_TYPE
        || op.type== OPERATOR_ASIN_TYPE
        || op.type== OPERATOR_ATAN_TYPE
        || op.type== OPERATOR_CEIL_TYPE
        || op.type== OPERATOR_COS_TYPE
        || op.type== OPERATOR_COSH_TYPE
        || op.type== OPERATOR_EXP_TYPE
        || op.type== OPERATOR_FABS_TYPE
        || op.type== OPERATOR_FLOOR_TYPE
        || op.type== OPERATOR_LOG_TYPE
        || op.type== OPERATOR_LOG10_TYPE
        || op.type== OPERATOR_SIN_TYPE
        || op.type== OPERATOR_SINH_TYPE
        || op.type== OPERATOR_SQRT_TYPE
        || op.type== OPERATOR_TAN_TYPE
        || op.type== OPERATOR_TANH_TYPE

        || op.type== OPERATOR_ELEMENT_POW_TYPE
        || op.type== OPERATOR_ELEMENT_FMAX_TYPE
        || op.type== OPERATOR_ELEMENT_FMIN_TYPE
        || op.type== OPERATOR_ELEMENT_MAX_TYPE
        || op.type== OPERATOR_ELEMENT_MIN_TYPE;

  }



}
//
filter_fun::filter_fun(pred_t pred, std::vector<size_t> & out) : pred_(pred), out_(out)
{ }

void filter_fun::operator()(isaac::math_expression const & math_expression, size_t root_idx, leaf_t leaf) const
{
  math_expression::node const * root_node = &math_expression.tree()[root_idx];
  if (leaf==PARENT_NODE_TYPE && pred_(*root_node))
    out_.push_back(root_idx);
}

//
std::vector<size_t> filter_nodes(bool (*pred)(math_expression::node const & node), isaac::math_expression const & math_expression, size_t root, bool inspect)
{
  std::vector<size_t> res;
  traverse(math_expression, root, filter_fun(pred, res), inspect);
  return res;
}

//
filter_elements_fun::filter_elements_fun(math_expression_node_subtype subtype, std::vector<lhs_rhs_element> & out) :
  subtype_(subtype), out_(out)
{ }

void filter_elements_fun::operator()(isaac::math_expression const & math_expression, size_t root_idx, leaf_t) const
{
  math_expression::node const * root_node = &math_expression.tree()[root_idx];
  if (root_node->lhs.subtype==subtype_)
    out_.push_back(root_node->lhs);
  if (root_node->rhs.subtype==subtype_)
    out_.push_back(root_node->rhs);
}


std::vector<lhs_rhs_element> filter_elements(math_expression_node_subtype subtype, isaac::math_expression const & math_expression)
{
  std::vector<lhs_rhs_element> res;
  traverse(math_expression, math_expression.root(), filter_elements_fun(subtype, res), true);
  return res;
}

/** @brief generate a string from an operation_node_type */
const char * evaluate(operation_node_type type)
{
  // unary expression
  switch (type)
  {
  //Function
  case OPERATOR_ABS_TYPE : return "abs";
  case OPERATOR_ACOS_TYPE : return "acos";
  case OPERATOR_ASIN_TYPE : return "asin";
  case OPERATOR_ATAN_TYPE : return "atan";
  case OPERATOR_CEIL_TYPE : return "ceil";
  case OPERATOR_COS_TYPE : return "cos";
  case OPERATOR_COSH_TYPE : return "cosh";
  case OPERATOR_EXP_TYPE : return "exp";
  case OPERATOR_FABS_TYPE : return "fabs";
  case OPERATOR_FLOOR_TYPE : return "floor";
  case OPERATOR_LOG_TYPE : return "log";
  case OPERATOR_LOG10_TYPE : return "log10";
  case OPERATOR_SIN_TYPE : return "sin";
  case OPERATOR_SINH_TYPE : return "sinh";
  case OPERATOR_SQRT_TYPE : return "sqrt";
  case OPERATOR_TAN_TYPE : return "tan";
  case OPERATOR_TANH_TYPE : return "tanh";

  case OPERATOR_ELEMENT_ARGFMAX_TYPE : return "argfmax";
  case OPERATOR_ELEMENT_ARGMAX_TYPE : return "argmax";
  case OPERATOR_ELEMENT_ARGFMIN_TYPE : return "argfmin";
  case OPERATOR_ELEMENT_ARGMIN_TYPE : return "argmin";
  case OPERATOR_ELEMENT_POW_TYPE : return "pow";

    //Arithmetic
  case OPERATOR_MINUS_TYPE : return "-";
  case OPERATOR_ASSIGN_TYPE : return "=";
  case OPERATOR_INPLACE_ADD_TYPE : return "+=";
  case OPERATOR_INPLACE_SUB_TYPE : return "-=";
  case OPERATOR_ADD_TYPE : return "+";
  case OPERATOR_SUB_TYPE : return "-";
  case OPERATOR_MULT_TYPE : return "*";
  case OPERATOR_ELEMENT_PROD_TYPE : return "*";
  case OPERATOR_DIV_TYPE : return "/";
  case OPERATOR_ELEMENT_DIV_TYPE : return "/";

    //Relational
  case OPERATOR_NEGATE_TYPE: return "!";
  case OPERATOR_ELEMENT_EQ_TYPE : return "==";
  case OPERATOR_ELEMENT_NEQ_TYPE : return "!=";
  case OPERATOR_ELEMENT_GREATER_TYPE : return ">";
  case OPERATOR_ELEMENT_GEQ_TYPE : return ">=";
  case OPERATOR_ELEMENT_LESS_TYPE : return "<";
  case OPERATOR_ELEMENT_LEQ_TYPE : return "<=";

  case OPERATOR_ELEMENT_FMAX_TYPE : return "fmax";
  case OPERATOR_ELEMENT_FMIN_TYPE : return "fmin";
  case OPERATOR_ELEMENT_MAX_TYPE : return "max";
  case OPERATOR_ELEMENT_MIN_TYPE : return "min";

    //Binary
  case OPERATOR_GEMM_NN_TYPE : return "prodNN";
  case OPERATOR_GEMM_TN_TYPE : return "prodTN";
  case OPERATOR_GEMM_NT_TYPE : return "prodNT";
  case OPERATOR_GEMM_TT_TYPE : return "prodTT";
  case OPERATOR_VDIAG_TYPE : return "vdiag";
  case OPERATOR_MATRIX_DIAG_TYPE : return "mdiag";
  case OPERATOR_MATRIX_ROW_TYPE : return "row";
  case OPERATOR_MATRIX_COLUMN_TYPE : return "col";
  case OPERATOR_PAIR_TYPE: return "pair";
  case OPERATOR_ACCESS_INDEX_TYPE: return "access";

    //FOR
  case OPERATOR_SFOR_TYPE: return "sfor";

  default : throw operation_not_supported_exception("Unsupported operator");
  }
}

evaluate_expression_traversal::evaluate_expression_traversal(std::map<std::string, std::string> const & accessors, std::string & str, mapping_type const & mapping) :
  accessors_(accessors), str_(str), mapping_(mapping)
{ }

void evaluate_expression_traversal::call_before_expansion(isaac::math_expression const & math_expression, std::size_t root_idx) const
{
  math_expression::node const & root_node = math_expression.tree()[root_idx];
  if(detail::is_cast(root_node.op))
    str_ += mapping_.at(std::make_pair(root_idx, PARENT_NODE_TYPE))->evaluate(accessors_);
  else if (( (root_node.op.type_family==OPERATOR_UNARY_TYPE_FAMILY&&root_node.op.type!=OPERATOR_ADD_TYPE) || detail::is_elementwise_function(root_node.op))
      && !detail::is_node_leaf(root_node.op))
    str_+=evaluate(root_node.op.type);
  if(root_node.op.type!=OPERATOR_FUSE)
    str_+="(";

}

void evaluate_expression_traversal::call_after_expansion(math_expression const & math_expression, std::size_t root_idx) const
{
  math_expression::node const & root_node = math_expression.tree()[root_idx];
  if(root_node.op.type!=OPERATOR_FUSE)
    str_+=")";
}

void evaluate_expression_traversal::operator()(isaac::math_expression const & math_expression, std::size_t root_idx, leaf_t leaf) const
{
  math_expression::node const & root_node = math_expression.tree()[root_idx];
  mapping_type::key_type key = std::make_pair(root_idx, leaf);
  if (leaf==PARENT_NODE_TYPE)
  {
    if (detail::is_node_leaf(root_node.op))
      str_ += mapping_.at(key)->evaluate(accessors_);
    else if(root_node.op.type_family!=OPERATOR_UNARY_TYPE_FAMILY)
    {
      if (detail::is_elementwise_operator(root_node.op))
        str_ += evaluate(root_node.op.type);
      else if (detail::is_elementwise_function(root_node.op))
        str_ += ",";
    }
  }
  else
  {
    if (leaf==LHS_NODE_TYPE)
    {
      if (root_node.lhs.type_family!=COMPOSITE_OPERATOR_FAMILY)
      {
        if (root_node.lhs.subtype==FOR_LOOP_INDEX_TYPE)
          str_ += "sforidx" + tools::to_string(root_node.lhs.for_idx.level);
        else
          str_ += mapping_.at(key)->evaluate(accessors_);
      }
    }

    if (leaf==RHS_NODE_TYPE)
    {
      if (root_node.rhs.type_family!=COMPOSITE_OPERATOR_FAMILY)
      {
        if (root_node.rhs.subtype==FOR_LOOP_INDEX_TYPE)
          str_ += "sforidx" + tools::to_string(root_node.rhs.for_idx.level);
        else
          str_ += mapping_.at(key)->evaluate(accessors_);
      }
    }
  }
}


std::string evaluate(leaf_t leaf, std::map<std::string, std::string> const & accessors,
                            isaac::math_expression const & math_expression, std::size_t root_idx, mapping_type const & mapping)
{
  std::string res;
  evaluate_expression_traversal traversal_functor(accessors, res, mapping);
  math_expression::node const & root_node = math_expression.tree()[root_idx];

  if (leaf==RHS_NODE_TYPE)
  {
    if (root_node.rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(math_expression, root_node.rhs.node_index, traversal_functor, false);
    else
      traversal_functor(math_expression, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(math_expression, root_node.lhs.node_index, traversal_functor, false);
    else
      traversal_functor(math_expression, root_idx, leaf);
  }
  else
    traverse(math_expression, root_idx, traversal_functor, false);

  return res;
}

void evaluate(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                     math_expression const & x, mapping_type const & mapping)
{
  stream << evaluate(leaf, accessors, x, x.root(), mapping) << std::endl;
}

process_traversal::process_traversal(std::map<std::string, std::string> const & accessors, kernel_generation_stream & stream,
                  mapping_type const & mapping, std::set<std::string> & already_processed) :
  accessors_(accessors),  stream_(stream), mapping_(mapping), already_processed_(already_processed)
{ }

void process_traversal::operator()(math_expression const & /*math_expression*/, std::size_t root_idx, leaf_t leaf) const
{
  mapping_type::const_iterator it = mapping_.find(std::make_pair(root_idx, leaf));
  if (it!=mapping_.end())
  {
    mapped_object * obj = it->second.get();
    std::string name = obj->name();

    if(accessors_.find(name)!=accessors_.end() && already_processed_.insert(name).second)
      for(std::map<std::string, std::string>::const_iterator itt = accessors_.lower_bound(name) ; itt != accessors_.upper_bound(name) ; ++itt)
        stream_ << obj->process(itt->second) << std::endl;

    std::string key = obj->type_key();
    if(accessors_.find(key)!=accessors_.end() && already_processed_.insert(name).second)
      for(std::map<std::string, std::string>::const_iterator itt = accessors_.lower_bound(key) ; itt != accessors_.upper_bound(key) ; ++itt)
        stream_ << obj->process(itt->second) << std::endl;
  }
}


void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                    isaac::math_expression const & math_expression, size_t root_idx, mapping_type const & mapping, std::set<std::string> & already_processed)
{
  process_traversal traversal_functor(accessors, stream, mapping, already_processed);
  math_expression::node const & root_node = math_expression.tree()[root_idx];

  if (leaf==RHS_NODE_TYPE)
  {
    if (root_node.rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(math_expression, root_node.rhs.node_index, traversal_functor, true);
    else
      traversal_functor(math_expression, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(math_expression, root_node.lhs.node_index, traversal_functor, true);
    else
      traversal_functor(math_expression, root_idx, leaf);
  }
  else
  {
    traverse(math_expression, root_idx, traversal_functor, true);
  }
}


void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                    math_expression const & x, mapping_type const & mapping)
{
  std::set<std::string> processed;
  process(stream, leaf, accessors, x, x.root(), mapping, processed);
}


void math_expression_representation_functor::append_id(char * & ptr, unsigned int val)
{
  if (val==0)
    *ptr++='0';
  else
    while (val>0)
    {
      *ptr++= (char)('0' + (val % 10));
      val /= 10;
    }
}

//void math_expression_representation_functor::append(driver::Buffer const & h, numeric_type dtype, char prefix, bool is_assigned) const
//{
//  *ptr_++=prefix;
//  *ptr_++=(char)dtype;
//  append_id(ptr_, binder_.get(h, is_assigned));
//}

void math_expression_representation_functor::append(lhs_rhs_element const & lhs_rhs, bool is_assigned) const
{
  if(lhs_rhs.subtype==DENSE_ARRAY_TYPE)
  {
      size4 shape = lhs_rhs.array->shape();
      char prefix;
      if(shape[0]==1 && shape[1]==1) prefix = '0';
      else if(shape[0]>1 && shape[1]==1) prefix = '1';
      else prefix = '2';
      numeric_type dtype = lhs_rhs.array->dtype();
      driver::Buffer const & data = lhs_rhs.array->data();

      *ptr_++=prefix;
      *ptr_++=(char)dtype;

      append_id(ptr_, binder_.get(data, is_assigned));
  }
}

math_expression_representation_functor::math_expression_representation_functor(symbolic_binder & binder, char *& ptr) : binder_(binder), ptr_(ptr){ }

void math_expression_representation_functor::append(char*& p, const char * str) const
{
  std::size_t n = std::strlen(str);
  std::memcpy(p, str, n);
  p+=n;
}

void math_expression_representation_functor::operator()(isaac::math_expression const & math_expression, std::size_t root_idx, leaf_t leaf_t) const
{
  math_expression::node const & root_node = math_expression.tree()[root_idx];
  if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    append(root_node.lhs, detail::is_assignment(root_node.op));
  else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    append(root_node.rhs, false);
  else if (leaf_t==PARENT_NODE_TYPE)
    append_id(ptr_,root_node.op.type);
}


}
