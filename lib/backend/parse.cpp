#include "atidlas/array.h"
#include "atidlas/backend/parse.h"
#include "atidlas/exception/operation_not_supported.h"

namespace atidlas
{

namespace detail
{

  bool is_node_leaf(op_element const & op)
  {
    return op.type==OPERATOR_TRANS_TYPE
        || op.type_family==OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY
        || op.type==OPERATOR_MATRIX_DIAG_TYPE
        || op.type==OPERATOR_VDIAG_TYPE
        || op.type==OPERATOR_REPEAT_TYPE
        || op.type==OPERATOR_MATRIX_ROW_TYPE
        || op.type==OPERATOR_MATRIX_COLUMN_TYPE
        || op.type_family==OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY
        || op.type_family==OPERATOR_ROWS_REDUCTION_TYPE_FAMILY
        || op.type_family==OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY
        || op.type==OPERATOR_OUTER_PROD_TYPE;
  }

  bool is_scalar_reduction(symbolic_expression_node const & node)
  {
    return node.op.type_family==OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY;
  }

  bool is_vector_reduction(symbolic_expression_node const & node)
  {
    return node.op.type_family==OPERATOR_ROWS_REDUCTION_TYPE_FAMILY
        || node.op.type_family==OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY;
  }

  bool is_elementwise_operator(op_element const & op)
  {
    return op.type== OPERATOR_ASSIGN_TYPE
        || op.type== OPERATOR_INPLACE_ADD_TYPE
        || op.type== OPERATOR_INPLACE_SUB_TYPE
        || op.type== OPERATOR_ADD_TYPE
        || op.type== OPERATOR_SUB_TYPE
        || op.type== OPERATOR_ELEMENT_PROD_TYPE
        || op.type== OPERATOR_ELEMENT_DIV_TYPE
        || op.type== OPERATOR_MULT_TYPE
        || op.type== OPERATOR_DIV_TYPE;
  }

  bool is_elementwise_function(op_element const & op)
  {
    return
        op.type == OPERATOR_CAST_CHAR_TYPE
        || op.type == OPERATOR_CAST_UCHAR_TYPE
        || op.type == OPERATOR_CAST_SHORT_TYPE
        || op.type == OPERATOR_CAST_USHORT_TYPE
        || op.type == OPERATOR_CAST_INT_TYPE
        || op.type == OPERATOR_CAST_UINT_TYPE
        || op.type == OPERATOR_CAST_LONG_TYPE
        || op.type == OPERATOR_CAST_ULONG_TYPE
        || op.type == OPERATOR_CAST_HALF_TYPE
        || op.type == OPERATOR_CAST_FLOAT_TYPE
        || op.type == OPERATOR_CAST_DOUBLE_TYPE

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
        || op.type== OPERATOR_ELEMENT_EQ_TYPE
        || op.type== OPERATOR_ELEMENT_NEQ_TYPE
        || op.type== OPERATOR_ELEMENT_GREATER_TYPE
        || op.type== OPERATOR_ELEMENT_LESS_TYPE
        || op.type== OPERATOR_ELEMENT_GEQ_TYPE
        || op.type== OPERATOR_ELEMENT_LEQ_TYPE
        || op.type== OPERATOR_ELEMENT_FMAX_TYPE
        || op.type== OPERATOR_ELEMENT_FMIN_TYPE
        || op.type== OPERATOR_ELEMENT_MAX_TYPE
        || op.type== OPERATOR_ELEMENT_MIN_TYPE;

  }

}
//
filter_fun::filter_fun(pred_t pred, std::vector<size_t> & out) : pred_(pred), out_(out)
{ }

void filter_fun::operator()(atidlas::symbolic_expression const & symbolic_expression, size_t root_idx, leaf_t) const
{
  symbolic_expression_node const * root_node = &symbolic_expression.tree()[root_idx];
  if (pred_(*root_node))
    out_.push_back(root_idx);
}

//
std::vector<size_t> filter_nodes(bool (*pred)(symbolic_expression_node const & node), atidlas::symbolic_expression const & symbolic_expression, bool inspect)
{
  std::vector<size_t> res;
  traverse(symbolic_expression, symbolic_expression.root(), filter_fun(pred, res), inspect);
  return res;
}

//
filter_elements_fun::filter_elements_fun(symbolic_expression_node_subtype subtype, std::vector<lhs_rhs_element> & out) :
  subtype_(subtype), out_(out)
{ }

void filter_elements_fun::operator()(atidlas::symbolic_expression const & symbolic_expression, size_t root_idx, leaf_t) const
{
  symbolic_expression_node const * root_node = &symbolic_expression.tree()[root_idx];
  if (root_node->lhs.subtype==subtype_)
    out_.push_back(root_node->lhs);
  if (root_node->rhs.subtype==subtype_)
    out_.push_back(root_node->rhs);
}


std::vector<lhs_rhs_element> filter_elements(symbolic_expression_node_subtype subtype, atidlas::symbolic_expression const & symbolic_expression)
{
  std::vector<lhs_rhs_element> res;
  traverse(symbolic_expression, symbolic_expression.root(), filter_elements_fun(subtype, res), true);
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

  case OPERATOR_CAST_CHAR_TYPE : return "(char)";
  case OPERATOR_CAST_UCHAR_TYPE : return "(uchar)";
  case OPERATOR_CAST_SHORT_TYPE : return "(short)";
  case OPERATOR_CAST_USHORT_TYPE : return "(ushort)";
  case OPERATOR_CAST_INT_TYPE : return "(int)";
  case OPERATOR_CAST_UINT_TYPE : return "(uint)";
  case OPERATOR_CAST_LONG_TYPE : return "(long)";
  case OPERATOR_CAST_ULONG_TYPE : return "(ulong)";
  case OPERATOR_CAST_HALF_TYPE : return "(half)";
  case OPERATOR_CAST_FLOAT_TYPE : return "(float)";
  case OPERATOR_CAST_DOUBLE_TYPE : return "(double)";

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
  case OPERATOR_ACCESS_TYPE : return "[]";

    //Relational
  case OPERATOR_ELEMENT_EQ_TYPE : return "isequal";
  case OPERATOR_ELEMENT_NEQ_TYPE : return "isnotequal";
  case OPERATOR_ELEMENT_GREATER_TYPE : return "isgreater";
  case OPERATOR_ELEMENT_GEQ_TYPE : return "isgreaterequal";
  case OPERATOR_ELEMENT_LESS_TYPE : return "isless";
  case OPERATOR_ELEMENT_LEQ_TYPE : return "islessequal";

  case OPERATOR_ELEMENT_FMAX_TYPE : return "fmax";
  case OPERATOR_ELEMENT_FMIN_TYPE : return "fmin";
  case OPERATOR_ELEMENT_MAX_TYPE : return "max";
  case OPERATOR_ELEMENT_MIN_TYPE : return "min";
    //Unary
  case OPERATOR_TRANS_TYPE : return "trans";

    //Binary
  case OPERATOR_MATRIX_PRODUCT_NN_TYPE : return "prodNN";
  case OPERATOR_MATRIX_PRODUCT_TN_TYPE : return "prodTN";
  case OPERATOR_MATRIX_PRODUCT_NT_TYPE : return "prodNT";
  case OPERATOR_MATRIX_PRODUCT_TT_TYPE : return "prodTT";
  case OPERATOR_VDIAG_TYPE : return "vdiag";
  case OPERATOR_MATRIX_DIAG_TYPE : return "mdiag";
  case OPERATOR_MATRIX_ROW_TYPE : return "row";
  case OPERATOR_MATRIX_COLUMN_TYPE : return "col";
  case OPERATOR_PAIR_TYPE: return "pair";

  default : throw operation_not_supported_exception("Unsupported operator");
  }
}

const char * operator_string(operation_node_type type)
{
  switch (type)
  {
  case OPERATOR_CAST_CHAR_TYPE : return "char";
  case OPERATOR_CAST_UCHAR_TYPE : return "uchar";
  case OPERATOR_CAST_SHORT_TYPE : return "short";
  case OPERATOR_CAST_USHORT_TYPE : return "ushort";
  case OPERATOR_CAST_INT_TYPE : return "int";
  case OPERATOR_CAST_UINT_TYPE : return "uint";
  case OPERATOR_CAST_LONG_TYPE : return "long";
  case OPERATOR_CAST_ULONG_TYPE : return "ulong";
  case OPERATOR_CAST_HALF_TYPE : return "half";
  case OPERATOR_CAST_FLOAT_TYPE : return "float";
  case OPERATOR_CAST_DOUBLE_TYPE : return "double";

  case OPERATOR_MINUS_TYPE : return "umin";
  case OPERATOR_ASSIGN_TYPE : return "assign";
  case OPERATOR_INPLACE_ADD_TYPE : return "ip_add";
  case OPERATOR_INPLACE_SUB_TYPE : return "ip_sub";
  case OPERATOR_ADD_TYPE : return "add";
  case OPERATOR_SUB_TYPE : return "sub";
  case OPERATOR_MULT_TYPE : return "mult";
  case OPERATOR_ELEMENT_PROD_TYPE : return "eprod";
  case OPERATOR_DIV_TYPE : return "div";
  case OPERATOR_ELEMENT_DIV_TYPE : return "ediv";
  case OPERATOR_ACCESS_TYPE : return "acc";
  default : return evaluate(type);
  }
}


evaluate_expression_traversal::evaluate_expression_traversal(std::map<std::string, std::string> const & accessors, std::string & str, mapping_type const & mapping) :
  accessors_(accessors), str_(str), mapping_(mapping)
{ }

void evaluate_expression_traversal::call_before_expansion(atidlas::symbolic_expression const & symbolic_expression, int_t root_idx) const
{
  symbolic_expression_node const & root_node = symbolic_expression.tree()[root_idx];
  if ((root_node.op.type_family==OPERATOR_UNARY_TYPE_FAMILY || detail::is_elementwise_function(root_node.op))
      && !detail::is_node_leaf(root_node.op))
    str_+=evaluate(root_node.op.type);
  str_+="(";

}

void evaluate_expression_traversal::call_after_expansion(symbolic_expression const & /*symbolic_expression*/, int_t /*root_idx*/) const
{
  str_+=")";
}

void evaluate_expression_traversal::operator()(atidlas::symbolic_expression const & symbolic_expression, int_t root_idx, leaf_t leaf) const
{
  symbolic_expression_node const & root_node = symbolic_expression.tree()[root_idx];
  mapping_type::key_type key = std::make_pair(root_idx, leaf);
  if (leaf==PARENT_NODE_TYPE)
  {
    if (detail::is_node_leaf(root_node.op))
      str_ += mapping_.at(key)->evaluate(accessors_);
    else if (detail::is_elementwise_operator(root_node.op))
      str_ += evaluate(root_node.op.type);
    else if (detail::is_elementwise_function(root_node.op) && root_node.op.type_family!=OPERATOR_UNARY_TYPE_FAMILY)
      str_ += ",";
  }
  else
  {
    if (leaf==LHS_NODE_TYPE)
    {
      if (root_node.lhs.type_family!=COMPOSITE_OPERATOR_FAMILY)
        str_ += mapping_.at(key)->evaluate(accessors_);
    }

    if (leaf==RHS_NODE_TYPE)
    {
      if (root_node.rhs.type_family!=COMPOSITE_OPERATOR_FAMILY)
        str_ += mapping_.at(key)->evaluate(accessors_);
    }
  }
}


std::string evaluate(leaf_t leaf, std::map<std::string, std::string> const & accessors,
                            atidlas::symbolic_expression const & symbolic_expression, int_t root_idx, mapping_type const & mapping)
{
  std::string res;
  evaluate_expression_traversal traversal_functor(accessors, res, mapping);
  symbolic_expression_node const & root_node = symbolic_expression.tree()[root_idx];

  if (leaf==RHS_NODE_TYPE)
  {
    if (root_node.rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(symbolic_expression, root_node.rhs.node_index, traversal_functor, false);
    else
      traversal_functor(symbolic_expression, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(symbolic_expression, root_node.lhs.node_index, traversal_functor, false);
    else
      traversal_functor(symbolic_expression, root_idx, leaf);
  }
  else
    traverse(symbolic_expression, root_idx, traversal_functor, false);

  return res;
}

void evaluate(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                     symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings)
{
  symbolic_expressions_container::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;

  for (mit = mappings.begin(), sit = symbolic_expressions.data().begin(); sit != symbolic_expressions.data().end(); ++mit, ++sit)
    stream << evaluate(leaf, accessors, **sit, (*sit)->root(), *mit) << ";" << std::endl;
}

process_traversal::process_traversal(std::map<std::string, std::string> const & accessors, kernel_generation_stream & stream,
                  mapping_type const & mapping, std::set<std::string> & already_processed) :
  accessors_(accessors),  stream_(stream), mapping_(mapping), already_processed_(already_processed)
{ }

void process_traversal::operator()(symbolic_expression const & /*symbolic_expression*/, int_t root_idx, leaf_t leaf) const
{
  mapping_type::const_iterator it = mapping_.find(std::make_pair(root_idx, leaf));
  if (it!=mapping_.end())
  {
    mapped_object * obj = it->second.get();
    std::string name = obj->name();

    if(accessors_.find(name)!=accessors_.end() && already_processed_.insert(name).second)
      for(std::map<std::string, std::string>::const_iterator itt = accessors_.lower_bound(name) ; itt != accessors_.upper_bound(name) ; ++itt)
      {
        stream_ << obj->process(itt->second) << std::endl;
      }

    std::string key = obj->type_key();
    if(accessors_.find(key)!=accessors_.end() && already_processed_.insert(name).second)
      for(std::map<std::string, std::string>::const_iterator itt = accessors_.lower_bound(key) ; itt != accessors_.upper_bound(key) ; ++itt)
      {
        stream_ << obj->process(itt->second) << std::endl;
      }
  }
}


void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                    atidlas::symbolic_expression const & symbolic_expression, size_t root_idx, mapping_type const & mapping, std::set<std::string> & already_processed)
{
  process_traversal traversal_functor(accessors, stream, mapping, already_processed);
  symbolic_expression_node const & root_node = symbolic_expression.tree()[root_idx];

  if (leaf==RHS_NODE_TYPE)
  {
    if (root_node.rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(symbolic_expression, root_node.rhs.node_index, traversal_functor, true);
    else
      traversal_functor(symbolic_expression, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(symbolic_expression, root_node.lhs.node_index, traversal_functor, true);
    else
      traversal_functor(symbolic_expression, root_idx, leaf);
  }
  else
  {
    traverse(symbolic_expression, root_idx, traversal_functor, true);
  }
}

void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                    symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings)
{
  symbolic_expressions_container::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;
  std::set<std::string> already_processed;

  for (mit = mappings.begin(), sit = symbolic_expressions.data().begin(); sit != symbolic_expressions.data().end(); ++mit, ++sit)
    process(stream, leaf, accessors, **sit, (*sit)->root(), *mit, already_processed);
}


void symbolic_expression_representation_functor::append_id(char * & ptr, unsigned int val)
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

void symbolic_expression_representation_functor::append(cl_mem h, numeric_type dtype, char prefix) const
{
  *ptr_++=prefix;
  *ptr_++=(char)dtype;
  append_id(ptr_, binder_.get(h));
}

void symbolic_expression_representation_functor::append(lhs_rhs_element const & lhs_rhs) const
{
  if(lhs_rhs.subtype==DENSE_ARRAY_TYPE)
    append(lhs_rhs.array.data, lhs_rhs.array.dtype, (char)(((int)'0')+((int)(lhs_rhs.array.shape1>1) + (int)(lhs_rhs.array.shape2>1))));
}

symbolic_expression_representation_functor::symbolic_expression_representation_functor(symbolic_binder & binder, char *& ptr) : binder_(binder), ptr_(ptr){ }

void symbolic_expression_representation_functor::append(char*& p, const char * str) const
{
  std::size_t n = std::strlen(str);
  std::memcpy(p, str, n);
  p+=n;
}

void symbolic_expression_representation_functor::operator()(atidlas::symbolic_expression const & symbolic_expression, int_t root_idx, leaf_t leaf_t) const
{
  symbolic_expression_node const & root_node = symbolic_expression.tree()[root_idx];
  if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    append(root_node.lhs);
  else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != COMPOSITE_OPERATOR_FAMILY)
    append(root_node.rhs);
  else if (leaf_t==PARENT_NODE_TYPE)
    append_id(ptr_,root_node.op.type);
}


}
