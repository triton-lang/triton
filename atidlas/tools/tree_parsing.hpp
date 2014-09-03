#ifndef ATIDLAS_TREE_PARSING_HPP
#define ATIDLAS_TREE_PARSING_HPP


#include <set>
#include "CL/cl.h"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "atidlas/mapped_objects.hpp"
#include "atidlas/tools/misc.hpp"
#include "atidlas/forwards.h"

namespace atidlas
{

namespace tools
{

/** @brief base functor class for traversing a statement */
class traversal_functor
{
public:
  void call_before_expansion(viennacl::scheduler::statement const &, atidlas_int_t) const { }
  void call_after_expansion(viennacl::scheduler::statement const &, atidlas_int_t) const { }
};

/** @brief Recursively execute a functor on a statement */
template<class Fun>
inline void traverse(viennacl::scheduler::statement const & statement, atidlas_int_t root_idx, Fun const & fun, bool inspect)
{
  viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];
  bool recurse = tools::node_leaf(root_node.op)?inspect:true;

  fun.call_before_expansion(statement, root_idx);

  //Lhs:
  if (recurse)
  {
    if (root_node.lhs.type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      traverse(statement, root_node.lhs.node_index, fun, inspect);
    if (root_node.lhs.type_family != viennacl::scheduler::INVALID_TYPE_FAMILY)
      fun(statement, root_idx, LHS_NODE_TYPE);
  }

  //Self:
  fun(statement, root_idx, PARENT_NODE_TYPE);

  //Rhs:
  if (recurse && root_node.rhs.type_family!=viennacl::scheduler::INVALID_TYPE_FAMILY)
  {
    if (root_node.rhs.type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      traverse(statement, root_node.rhs.node_index, fun, inspect);
    if (root_node.rhs.type_family != viennacl::scheduler::INVALID_TYPE_FAMILY)
      fun(statement, root_idx, RHS_NODE_TYPE);
  }

  fun.call_after_expansion(statement, root_idx);
}

class filter_fun : public traversal_functor
{
public:
  typedef bool (*pred_t)(viennacl::scheduler::statement_node const & node);

  filter_fun(pred_t pred, std::vector<size_t> & out) : pred_(pred), out_(out){ }

  void operator()(viennacl::scheduler::statement const & statement, size_t root_idx, leaf_t) const
  {
    viennacl::scheduler::statement_node const * root_node = &statement.array()[root_idx];
    if (pred_(*root_node))
      out_.push_back(root_idx);
  }
private:
  pred_t pred_;
  std::vector<size_t> & out_;
};

inline std::vector<size_t> filter_nodes(bool (*pred)(viennacl::scheduler::statement_node const & node), viennacl::scheduler::statement const & statement, bool inspect)
{
  std::vector<size_t> res;
  tools::traverse(statement, statement.root(), filter_fun(pred, res), inspect);
  return res;
}

class filter_elements_fun : public traversal_functor
{
public:
  filter_elements_fun(viennacl::scheduler::statement_node_subtype subtype, std::vector<viennacl::scheduler::lhs_rhs_element> & out) : subtype_(subtype), out_(out) { }

  void operator()(viennacl::scheduler::statement const & statement, size_t root_idx, leaf_t) const
  {
    viennacl::scheduler::statement_node const * root_node = &statement.array()[root_idx];
    if (root_node->lhs.subtype==subtype_)
      out_.push_back(root_node->lhs);
    if (root_node->rhs.subtype==subtype_)
      out_.push_back(root_node->rhs);
  }
private:
  viennacl::scheduler::statement_node_subtype subtype_;
  std::vector<viennacl::scheduler::lhs_rhs_element> & out_;
};

inline std::vector<viennacl::scheduler::lhs_rhs_element> filter_elements(viennacl::scheduler::statement_node_subtype subtype, viennacl::scheduler::statement const & statement)
{
  std::vector<viennacl::scheduler::lhs_rhs_element> res;
  tools::traverse(statement, statement.root(), filter_elements_fun(subtype, res), true);
  return res;
}

/** @brief generate a string from an operation_node_type */
inline const char * evaluate(viennacl::scheduler::operation_node_type type)
{
  using namespace viennacl::scheduler;
  // unary expression
  switch (type)
  {
  //Function
  case OPERATION_UNARY_ABS_TYPE : return "abs";
  case OPERATION_UNARY_ACOS_TYPE : return "acos";
  case OPERATION_UNARY_ASIN_TYPE : return "asin";
  case OPERATION_UNARY_ATAN_TYPE : return "atan";
  case OPERATION_UNARY_CEIL_TYPE : return "ceil";
  case OPERATION_UNARY_COS_TYPE : return "cos";
  case OPERATION_UNARY_COSH_TYPE : return "cosh";
  case OPERATION_UNARY_EXP_TYPE : return "exp";
  case OPERATION_UNARY_FABS_TYPE : return "fabs";
  case OPERATION_UNARY_FLOOR_TYPE : return "floor";
  case OPERATION_UNARY_LOG_TYPE : return "log";
  case OPERATION_UNARY_LOG10_TYPE : return "log10";
  case OPERATION_UNARY_SIN_TYPE : return "sin";
  case OPERATION_UNARY_SINH_TYPE : return "sinh";
  case OPERATION_UNARY_SQRT_TYPE : return "sqrt";
  case OPERATION_UNARY_TAN_TYPE : return "tan";
  case OPERATION_UNARY_TANH_TYPE : return "tanh";

  case OPERATION_UNARY_CAST_CHAR_TYPE : return "(char)";
  case OPERATION_UNARY_CAST_UCHAR_TYPE : return "(uchar)";
  case OPERATION_UNARY_CAST_SHORT_TYPE : return "(short)";
  case OPERATION_UNARY_CAST_USHORT_TYPE : return "(ushort)";
  case OPERATION_UNARY_CAST_INT_TYPE : return "(int)";
  case OPERATION_UNARY_CAST_UINT_TYPE : return "(uint)";
  case OPERATION_UNARY_CAST_LONG_TYPE : return "(long)";
  case OPERATION_UNARY_CAST_ULONG_TYPE : return "(ulong)";
  case OPERATION_UNARY_CAST_HALF_TYPE : return "(half)";
  case OPERATION_UNARY_CAST_FLOAT_TYPE : return "(float)";
  case OPERATION_UNARY_CAST_DOUBLE_TYPE : return "(double)";

  case OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE : return "argfmax";
  case OPERATION_BINARY_ELEMENT_ARGMAX_TYPE : return "argmax";
  case OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE : return "argfmin";
  case OPERATION_BINARY_ELEMENT_ARGMIN_TYPE : return "argmin";
  case OPERATION_BINARY_ELEMENT_POW_TYPE : return "pow";

    //Arithmetic
  case OPERATION_UNARY_MINUS_TYPE : return "-";
  case OPERATION_BINARY_ASSIGN_TYPE : return "=";
  case OPERATION_BINARY_INPLACE_ADD_TYPE : return "+=";
  case OPERATION_BINARY_INPLACE_SUB_TYPE : return "-=";
  case OPERATION_BINARY_ADD_TYPE : return "+";
  case OPERATION_BINARY_SUB_TYPE : return "-";
  case OPERATION_BINARY_MULT_TYPE : return "*";
  case OPERATION_BINARY_ELEMENT_PROD_TYPE : return "*";
  case OPERATION_BINARY_DIV_TYPE : return "/";
  case OPERATION_BINARY_ELEMENT_DIV_TYPE : return "/";
  case OPERATION_BINARY_ACCESS_TYPE : return "[]";

    //Relational
  case OPERATION_BINARY_ELEMENT_EQ_TYPE : return "isequal";
  case OPERATION_BINARY_ELEMENT_NEQ_TYPE : return "isnotequal";
  case OPERATION_BINARY_ELEMENT_GREATER_TYPE : return "isgreater";
  case OPERATION_BINARY_ELEMENT_GEQ_TYPE : return "isgreaterequal";
  case OPERATION_BINARY_ELEMENT_LESS_TYPE : return "isless";
  case OPERATION_BINARY_ELEMENT_LEQ_TYPE : return "islessequal";

  case OPERATION_BINARY_ELEMENT_FMAX_TYPE : return "fmax";
  case OPERATION_BINARY_ELEMENT_FMIN_TYPE : return "fmin";
  case OPERATION_BINARY_ELEMENT_MAX_TYPE : return "max";
  case OPERATION_BINARY_ELEMENT_MIN_TYPE : return "min";
    //Unary
  case OPERATION_UNARY_TRANS_TYPE : return "trans";

    //Binary
  case OPERATION_BINARY_INNER_PROD_TYPE : return "iprod";
  case OPERATION_BINARY_MAT_MAT_PROD_TYPE : return "mmprod";
  case OPERATION_BINARY_MAT_VEC_PROD_TYPE : return "mvprod";
  case OPERATION_BINARY_VECTOR_DIAG_TYPE : return "vdiag";
  case OPERATION_BINARY_MATRIX_DIAG_TYPE : return "mdiag";
  case OPERATION_BINARY_MATRIX_ROW_TYPE : return "row";
  case OPERATION_BINARY_MATRIX_COLUMN_TYPE : return "col";

  default : throw generator_not_supported_exception("Unsupported operator");
  }
}

inline const char * operator_string(viennacl::scheduler::operation_node_type type)
{
  using namespace viennacl::scheduler;
  switch (type)
  {
  case OPERATION_UNARY_CAST_CHAR_TYPE : return "char";
  case OPERATION_UNARY_CAST_UCHAR_TYPE : return "uchar";
  case OPERATION_UNARY_CAST_SHORT_TYPE : return "short";
  case OPERATION_UNARY_CAST_USHORT_TYPE : return "ushort";
  case OPERATION_UNARY_CAST_INT_TYPE : return "int";
  case OPERATION_UNARY_CAST_UINT_TYPE : return "uint";
  case OPERATION_UNARY_CAST_LONG_TYPE : return "long";
  case OPERATION_UNARY_CAST_ULONG_TYPE : return "ulong";
  case OPERATION_UNARY_CAST_HALF_TYPE : return "half";
  case OPERATION_UNARY_CAST_FLOAT_TYPE : return "float";
  case OPERATION_UNARY_CAST_DOUBLE_TYPE : return "double";

  case OPERATION_UNARY_MINUS_TYPE : return "umin";
  case OPERATION_BINARY_ASSIGN_TYPE : return "assign";
  case OPERATION_BINARY_INPLACE_ADD_TYPE : return "ip_add";
  case OPERATION_BINARY_INPLACE_SUB_TYPE : return "ip_sub";
  case OPERATION_BINARY_ADD_TYPE : return "add";
  case OPERATION_BINARY_SUB_TYPE : return "sub";
  case OPERATION_BINARY_MULT_TYPE : return "mult";
  case OPERATION_BINARY_ELEMENT_PROD_TYPE : return "eprod";
  case OPERATION_BINARY_DIV_TYPE : return "div";
  case OPERATION_BINARY_ELEMENT_DIV_TYPE : return "ediv";
  case OPERATION_BINARY_ACCESS_TYPE : return "acc";
  default : return evaluate(type);
  }
}

/** @brief functor for generating the expression string from a statement */
class evaluate_expression_traversal: public tools::traversal_functor
{
private:
  std::map<std::string, std::string> const & accessors_;
  std::string & str_;
  mapping_type const & mapping_;

public:
  evaluate_expression_traversal(std::map<std::string, std::string> const & accessors, std::string & str, mapping_type const & mapping) : accessors_(accessors), str_(str), mapping_(mapping){ }

  void call_before_expansion(viennacl::scheduler::statement const & statement, atidlas_int_t root_idx) const
  {
    viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];
    if ((root_node.op.type_family==viennacl::scheduler::OPERATION_UNARY_TYPE_FAMILY || tools::elementwise_function(root_node.op))
        && !tools::node_leaf(root_node.op))
      str_+=tools::evaluate(root_node.op.type);
    str_+="(";

  }

  void call_after_expansion(viennacl::scheduler::statement const & /*statement*/, atidlas_int_t /*root_idx*/) const
  {
    str_+=")";
  }

  void operator()(viennacl::scheduler::statement const & statement, atidlas_int_t root_idx, leaf_t leaf) const
  {
    viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];
    mapping_type::key_type key = std::make_pair(root_idx, leaf);
    if (leaf==PARENT_NODE_TYPE)
    {
      if (tools::node_leaf(root_node.op))
        str_ += mapping_.at(key)->evaluate(accessors_);
      else if (tools::elementwise_operator(root_node.op))
        str_ += tools::evaluate(root_node.op.type);
      else if (root_node.op.type_family!=viennacl::scheduler::OPERATION_UNARY_TYPE_FAMILY && tools::elementwise_function(root_node.op))
        str_ += ",";
    }
    else
    {
      if (leaf==LHS_NODE_TYPE)
      {
        if (root_node.lhs.type_family!=viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
          str_ += mapping_.at(key)->evaluate(accessors_);
      }

      if (leaf==RHS_NODE_TYPE)
      {
        if (root_node.rhs.type_family!=viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
          str_ += mapping_.at(key)->evaluate(accessors_);
      }
    }
  }
};

inline std::string evaluate(leaf_t leaf, std::map<std::string, std::string> const & accessors,
                            viennacl::scheduler::statement const & statement, atidlas_int_t root_idx, mapping_type const & mapping)
{
  std::string res;
  evaluate_expression_traversal traversal_functor(accessors, res, mapping);
  viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];

  if (leaf==RHS_NODE_TYPE)
  {
    if (root_node.rhs.type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      tools::traverse(statement, root_node.rhs.node_index, traversal_functor, false);
    else
      traversal_functor(statement, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      tools::traverse(statement, root_node.lhs.node_index, traversal_functor, false);
    else
      traversal_functor(statement, root_idx, leaf);
  }
  else
    tools::traverse(statement, root_idx, traversal_functor, false);

  return res;
}

inline void evaluate(tools::kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                     statements_container const & statements, std::vector<mapping_type> const & mappings)
{
  statements_container::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;

  for (mit = mappings.begin(), sit = statements.data().begin(); sit != statements.data().end(); ++mit, ++sit)
    stream << evaluate(leaf, accessors, *sit, sit->root(), *mit) << ";" << std::endl;
}


/** @brief functor for fetching or writing-back the elements in a statement */
class process_traversal : public tools::traversal_functor
{
public:
  process_traversal(std::multimap<std::string, std::string> const & accessors, tools::kernel_generation_stream & stream,
                    mapping_type const & mapping, std::set<std::string> & already_processed) : accessors_(accessors),  stream_(stream), mapping_(mapping), already_processed_(already_processed){ }

  void operator()(viennacl::scheduler::statement const & /*statement*/, atidlas_int_t root_idx, leaf_t leaf) const
  {
    mapping_type::const_iterator it = mapping_.find(std::make_pair(root_idx, leaf));
    if (it!=mapping_.end())
    {
      mapped_object * obj = it->second.get();
      std::string name = obj->name();
      if(accessors_.find(name)!=accessors_.end() && already_processed_.insert(obj->process("#name")).second)
        for(std::multimap<std::string, std::string>::const_iterator it = accessors_.lower_bound(name) ; it != accessors_.upper_bound(name) ; ++it)
          stream_ << obj->process(it->second) << std::endl;

      std::string key = obj->type_key();
      if(accessors_.find(key)!=accessors_.end() && already_processed_.insert(obj->process("#name")).second)
        for(std::multimap<std::string, std::string>::const_iterator it = accessors_.lower_bound(key) ; it != accessors_.upper_bound(key) ; ++it)
          stream_ << obj->process(it->second) << std::endl;
    }
  }

private:
  std::multimap<std::string, std::string> accessors_;
  tools::kernel_generation_stream & stream_;
  mapping_type const & mapping_;
  std::set<std::string> & already_processed_;
};

inline void process(tools::kernel_generation_stream & stream, leaf_t leaf, std::multimap<std::string, std::string> const & accessors,
                    viennacl::scheduler::statement const & statement, size_t root_idx, mapping_type const & mapping, std::set<std::string> & already_processed)
{
  process_traversal traversal_functor(accessors, stream, mapping, already_processed);
  viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];

  if (leaf==RHS_NODE_TYPE)
  {
    if (root_node.rhs.type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      tools::traverse(statement, root_node.rhs.node_index, traversal_functor, true);
    else
      traversal_functor(statement, root_idx, leaf);
  }
  else if (leaf==LHS_NODE_TYPE)
  {
    if (root_node.lhs.type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      tools::traverse(statement, root_node.lhs.node_index, traversal_functor, true);
    else
      traversal_functor(statement, root_idx, leaf);
  }
  else
  {
    tools::traverse(statement, root_idx, traversal_functor, true);
  }
}

inline void process(tools::kernel_generation_stream & stream, leaf_t leaf, std::multimap<std::string, std::string> const & accessors,
                    statements_container const & statements, std::vector<mapping_type> const & mappings)
{
  statements_container::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;
  std::set<std::string> already_processed;

  for (mit = mappings.begin(), sit = statements.data().begin(); sit != statements.data().end(); ++mit, ++sit)
    process(stream, leaf, accessors, *sit, sit->root(), *mit, already_processed);
}


class statement_representation_functor : public traversal_functor{
private:
  static void append_id(char * & ptr, unsigned int val)
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

public:
  typedef void result_type;

  statement_representation_functor(symbolic_binder & binder, char *& ptr) : binder_(binder), ptr_(ptr){ }

  template<class NumericT>
  inline result_type operator()(NumericT const & /*scal*/) const
  {
    *ptr_++='h'; //host
    *ptr_++='s'; //scalar
    *ptr_++=tools::first_letter_of_type<NumericT>::value();
  }

  /** @brief Scalar mapping */
  template<class NumericT>
  inline result_type operator()(viennacl::scalar<NumericT> const & scal) const
  {
    *ptr_++='s'; //scalar
    *ptr_++=tools::first_letter_of_type<NumericT>::value();
    append_id(ptr_, binder_.get(&viennacl::traits::handle(scal)));
  }

  /** @brief Vector mapping */
  template<class NumericT>
  inline result_type operator()(viennacl::vector_base<NumericT> const & vec) const
  {
    *ptr_++='v'; //vector
    *ptr_++=tools::first_letter_of_type<NumericT>::value();
    append_id(ptr_, binder_.get(&viennacl::traits::handle(vec)));
  }

  /** @brief Implicit vector mapping */
  template<class NumericT>
  inline result_type operator()(viennacl::implicit_vector_base<NumericT> const & /*vec*/) const
  {
    *ptr_++='i'; //implicit
    *ptr_++='v'; //vector
    *ptr_++=tools::first_letter_of_type<NumericT>::value();
  }

  /** @brief Matrix mapping */
  template<class NumericT>
  inline result_type operator()(viennacl::matrix_base<NumericT> const & mat) const
  {
    *ptr_++='m'; //Matrix
    *ptr_++=tools::first_letter_of_type<NumericT>::value();
    append_id(ptr_, binder_.get(&viennacl::traits::handle(mat)));
  }

  /** @brief Implicit matrix mapping */
  template<class NumericT>
  inline result_type operator()(viennacl::implicit_matrix_base<NumericT> const & /*mat*/) const
  {
    *ptr_++='i'; //implicit
    *ptr_++='m'; //matrix
    *ptr_++=tools::first_letter_of_type<NumericT>::value();
  }

  static inline void append(char*& p, const char * str)
  {
    std::size_t n = std::strlen(str);
    std::memcpy(p, str, n);
    p+=n;
  }

  inline void operator()(viennacl::scheduler::statement const & statement, atidlas_int_t root_idx, leaf_t leaf_t) const
  {
    viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];
    if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      tools::call_on_element(root_node.lhs, *this);
    else if (root_node.op.type_family==viennacl::scheduler::OPERATION_BINARY_TYPE_FAMILY && leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
      tools::call_on_element(root_node.rhs, *this);
    else if (leaf_t==PARENT_NODE_TYPE)
      append_id(ptr_,root_node.op.type);
  }

private:
  symbolic_binder & binder_;
  char *& ptr_;
};

inline std::string statements_representation(statements_container const & statements, binding_policy_t binding_policy)
{
  std::vector<char> program_name_vector(256);
  char* program_name = program_name_vector.data();
  if (statements.order()==statements_container::INDEPENDENT)
    *program_name++='i';
  else
    *program_name++='s';
  tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy);
  for (statements_container::data_type::const_iterator it = statements.data().begin(); it != statements.data().end(); ++it)
    tools::traverse(*it, it->root(), tools::statement_representation_functor(*binder, program_name),true);
  *program_name='\0';
  return std::string(program_name_vector.data());
}

}
}
#endif
