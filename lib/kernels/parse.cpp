#include <cstring>

#include "to_string.hpp"

#include "isaac/array.h"
#include "isaac/kernels/parse.h"
#include "isaac/exception/operation_not_supported.h"

namespace isaac
{

namespace detail
{



  bool is_scalar_reduce_1d(math_expression::node const & node)
  {
    return node.op.type_family==VECTOR_DOT_TYPE_FAMILY;
  }

  bool is_vector_reduce_1d(math_expression::node const & node)
  {
    return node.op.type_family==ROWS_DOT_TYPE_FAMILY
        || node.op.type_family==COLUMNS_DOT_TYPE_FAMILY;
  }

  bool is_assignment(op_element const & op)
  {
      return op.type== ASSIGN_TYPE
              || op.type== INPLACE_ADD_TYPE
              || op.type== INPLACE_SUB_TYPE;
  }

  bool is_elementwise_operator(op_element const & op)
  {
    return is_assignment(op)
        || op.type== ADD_TYPE
        || op.type== SUB_TYPE
        || op.type== ELEMENT_PROD_TYPE
        || op.type== ELEMENT_DIV_TYPE
        || op.type== MULT_TYPE
        || op.type== DIV_TYPE
        || op.type== ELEMENT_EQ_TYPE
        || op.type== ELEMENT_NEQ_TYPE
        || op.type== ELEMENT_GREATER_TYPE
        || op.type== ELEMENT_LESS_TYPE
        || op.type== ELEMENT_GEQ_TYPE
        || op.type== ELEMENT_LEQ_TYPE ;
  }

  bool bypass(op_element const & op)
  {
        return op.type == RESHAPE_TYPE
             ||op.type == TRANS_TYPE;
  }

  bool is_cast(op_element const & op)
  {
        return op.type== CAST_BOOL_TYPE
            || op.type== CAST_CHAR_TYPE
            || op.type== CAST_UCHAR_TYPE
            || op.type== CAST_SHORT_TYPE
            || op.type== CAST_USHORT_TYPE
            || op.type== CAST_INT_TYPE
            || op.type== CAST_UINT_TYPE
            || op.type== CAST_LONG_TYPE
            || op.type== CAST_ULONG_TYPE
            || op.type== CAST_FLOAT_TYPE
            || op.type== CAST_DOUBLE_TYPE
            ;
  }

  bool is_node_leaf(op_element const & op)
  {
    return op.type==MATRIX_DIAG_TYPE
        || op.type==VDIAG_TYPE
        || op.type==REPEAT_TYPE
        || op.type==MATRIX_ROW_TYPE
        || op.type==MATRIX_COLUMN_TYPE
        || op.type==ACCESS_INDEX_TYPE
        || op.type==OUTER_PROD_TYPE
        || op.type_family==VECTOR_DOT_TYPE_FAMILY
        || op.type_family==ROWS_DOT_TYPE_FAMILY
        || op.type_family==COLUMNS_DOT_TYPE_FAMILY
        || op.type_family==MATRIX_PRODUCT_TYPE_FAMILY
        ;
  }

  bool is_elementwise_function(op_element const & op)
  {
    return is_cast(op)
        || op.type== ABS_TYPE
        || op.type== ACOS_TYPE
        || op.type== ASIN_TYPE
        || op.type== ATAN_TYPE
        || op.type== CEIL_TYPE
        || op.type== COS_TYPE
        || op.type== COSH_TYPE
        || op.type== EXP_TYPE
        || op.type== FABS_TYPE
        || op.type== FLOOR_TYPE
        || op.type== LOG_TYPE
        || op.type== LOG10_TYPE
        || op.type== SIN_TYPE
        || op.type== SINH_TYPE
        || op.type== SQRT_TYPE
        || op.type== TAN_TYPE
        || op.type== TANH_TYPE

        || op.type== ELEMENT_POW_TYPE
        || op.type== ELEMENT_FMAX_TYPE
        || op.type== ELEMENT_FMIN_TYPE
        || op.type== ELEMENT_MAX_TYPE
        || op.type== ELEMENT_MIN_TYPE;

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
filter_elements_fun::filter_elements_fun(node_type subtype, std::vector<tree_node> & out) :
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


std::vector<tree_node> filter_elements(node_type subtype, isaac::math_expression const & math_expression)
{
  std::vector<tree_node> res;
  traverse(math_expression, math_expression.root(), filter_elements_fun(subtype, res), true);
  return res;
}

/** @brief generate a string from an operation_type */
const char * evaluate(operation_type type)
{
  // unary expression
  switch (type)
  {
  //Function
  case ABS_TYPE : return "abs";
  case ACOS_TYPE : return "acos";
  case ASIN_TYPE : return "asin";
  case ATAN_TYPE : return "atan";
  case CEIL_TYPE : return "ceil";
  case COS_TYPE : return "cos";
  case COSH_TYPE : return "cosh";
  case EXP_TYPE : return "exp";
  case FABS_TYPE : return "fabs";
  case FLOOR_TYPE : return "floor";
  case LOG_TYPE : return "log";
  case LOG10_TYPE : return "log10";
  case SIN_TYPE : return "sin";
  case SINH_TYPE : return "sinh";
  case SQRT_TYPE : return "sqrt";
  case TAN_TYPE : return "tan";
  case TANH_TYPE : return "tanh";

  case ELEMENT_ARGFMAX_TYPE : return "argfmax";
  case ELEMENT_ARGMAX_TYPE : return "argmax";
  case ELEMENT_ARGFMIN_TYPE : return "argfmin";
  case ELEMENT_ARGMIN_TYPE : return "argmin";
  case ELEMENT_POW_TYPE : return "pow";

  //Arithmetic
  case MINUS_TYPE : return "-";
  case ASSIGN_TYPE : return "=";
  case INPLACE_ADD_TYPE : return "+=";
  case INPLACE_SUB_TYPE : return "-=";
  case ADD_TYPE : return "+";
  case SUB_TYPE : return "-";
  case MULT_TYPE : return "*";
  case ELEMENT_PROD_TYPE : return "*";
  case DIV_TYPE : return "/";
  case ELEMENT_DIV_TYPE : return "/";

  //Relational
  case NEGATE_TYPE: return "!";
  case ELEMENT_EQ_TYPE : return "==";
  case ELEMENT_NEQ_TYPE : return "!=";
  case ELEMENT_GREATER_TYPE : return ">";
  case ELEMENT_GEQ_TYPE : return ">=";
  case ELEMENT_LESS_TYPE : return "<";
  case ELEMENT_LEQ_TYPE : return "<=";

  case ELEMENT_FMAX_TYPE : return "fmax";
  case ELEMENT_FMIN_TYPE : return "fmin";
  case ELEMENT_MAX_TYPE : return "max";
  case ELEMENT_MIN_TYPE : return "min";

  //Binary
  case MATRIX_PRODUCT_NN_TYPE : return "prodNN";
  case MATRIX_PRODUCT_TN_TYPE : return "prodTN";
  case MATRIX_PRODUCT_NT_TYPE : return "prodNT";
  case MATRIX_PRODUCT_TT_TYPE : return "prodTT";
  case VDIAG_TYPE : return "vdiag";
  case MATRIX_DIAG_TYPE : return "mdiag";
  case MATRIX_ROW_TYPE : return "row";
  case MATRIX_COLUMN_TYPE : return "col";
  case PAIR_TYPE: return "pair";
  case ACCESS_INDEX_TYPE: return "access";

  //FOR
  case SFOR_TYPE: return "sfor";

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
  else if (( (root_node.op.type_family==UNARY_TYPE_FAMILY&&root_node.op.type!=ADD_TYPE) || detail::is_elementwise_function(root_node.op))
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
    else if(root_node.op.type_family!=UNARY_TYPE_FAMILY)
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
      if (root_node.lhs.subtype!=COMPOSITE_OPERATOR_TYPE)
      {
        if (root_node.lhs.subtype==FOR_LOOP_INDEX_TYPE)
          str_ += "sforidx" + tools::to_string(root_node.lhs.for_idx.level);
        else
          str_ += mapping_.at(key)->evaluate(accessors_);
      }
    }

    if (leaf==RHS_NODE_TYPE)
    {
      if (root_node.rhs.subtype!=COMPOSITE_OPERATOR_TYPE)
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
    if (root_node.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
      traverse(math_expression, root_node.rhs.node_index, traversal_functor, false);
    else
      traversal_functor(math_expression, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.subtype==COMPOSITE_OPERATOR_TYPE)
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
    if (root_node.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
      traverse(math_expression, root_node.rhs.node_index, traversal_functor, true);
    else
      traversal_functor(math_expression, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.subtype==COMPOSITE_OPERATOR_TYPE)
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

void math_expression_representation_functor::append(tree_node const & lhs_rhs, bool is_assigned) const
{
  if(lhs_rhs.subtype==DENSE_ARRAY_TYPE)
  {
      for(int i = 0 ; i < lhs_rhs.array->dim() ; ++i)
        *ptr_++= lhs_rhs.array->shape()[i]>1?'n':'1';
      numeric_type dtype = lhs_rhs.array->dtype();
      *ptr_++=(char)dtype;

      append_id(ptr_, binder_.get(lhs_rhs.array, is_assigned));
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
  if (leaf_t==LHS_NODE_TYPE && root_node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
    append(root_node.lhs, detail::is_assignment(root_node.op));
  else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
    append(root_node.rhs, false);
  else if (leaf_t==PARENT_NODE_TYPE)
    append_id(ptr_,root_node.op.type);
}


}
