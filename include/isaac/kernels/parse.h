#ifndef ISAAC_BACKEND_PARSE_H
#define ISAAC_BACKEND_PARSE_H

#include <set>
#include "isaac/kernels/mapped_object.h"
#include "isaac/kernels/binder.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

namespace detail
{

  bool is_node_leaf(op_element const & op);
  bool is_scalar_reduce_1d(math_expression::node const & node);
  bool is_vector_reduce_1d(math_expression::node const & node);
  bool is_assignment(op_element const & op);
  bool is_elementwise_operator(op_element const & op);
  bool is_elementwise_function(op_element const & op);
  bool is_cast(op_element const & op);
  bool bypass(op_element const & op);
}

class scalar;

/** @brief base functor class for traversing a math_expression */
class traversal_functor
{
public:
  void call_before_expansion(math_expression const &, std::size_t) const { }
  void call_after_expansion(math_expression const &, std::size_t) const { }
};


/** @brief Recursively execute a functor on a math_expression */
template<class Fun>
inline void traverse(isaac::math_expression const & math_expression, std::size_t root_idx, Fun const & fun, bool inspect)
{
  math_expression::node const & root_node = math_expression.tree()[root_idx];
  bool recurse = detail::is_node_leaf(root_node.op)?inspect:true;
  bool bypass = detail::bypass(root_node.op);

  if(!bypass)
    fun.call_before_expansion(math_expression, root_idx);

  //Lhs:
  if (recurse)
  {
    if (root_node.lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(math_expression, root_node.lhs.node_index, fun, inspect);
    if (root_node.lhs.type_family != INVALID_TYPE_FAMILY)
      fun(math_expression, root_idx, LHS_NODE_TYPE);
  }

  //Self:
  if(!bypass)
    fun(math_expression, root_idx, PARENT_NODE_TYPE);

  //Rhs:
  if (recurse && root_node.rhs.type_family!=INVALID_TYPE_FAMILY)
  {
    if (root_node.rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
      traverse(math_expression, root_node.rhs.node_index, fun, inspect);
    if (root_node.rhs.type_family != INVALID_TYPE_FAMILY)
      fun(math_expression, root_idx, RHS_NODE_TYPE);
  }

  if(!bypass)
    fun.call_after_expansion(math_expression, root_idx);
}

class filter_fun : public traversal_functor
{
public:
  typedef bool (*pred_t)(math_expression::node const & node);
  filter_fun(pred_t pred, std::vector<size_t> & out);
  void operator()(isaac::math_expression const & math_expression, size_t root_idx, leaf_t) const;
private:
  pred_t pred_;
  std::vector<size_t> & out_;
};

class filter_elements_fun : public traversal_functor
{
public:
  filter_elements_fun(math_expression_node_subtype subtype, std::vector<lhs_rhs_element> & out);
  void operator()(isaac::math_expression const & math_expression, size_t root_idx, leaf_t) const;
private:
  math_expression_node_subtype subtype_;
  std::vector<lhs_rhs_element> & out_;
};

std::vector<size_t> filter_nodes(bool (*pred)(math_expression::node const & node),
                                        isaac::math_expression const & math_expression,
                                        size_t root,
                                        bool inspect);

std::vector<lhs_rhs_element> filter_elements(math_expression_node_subtype subtype,
                                             isaac::math_expression const & math_expression);
const char * evaluate(operation_node_type type);

/** @brief functor for generating the expression string from a math_expression */
class evaluate_expression_traversal: public traversal_functor
{
private:
  std::map<std::string, std::string> const & accessors_;
  std::string & str_;
  mapping_type const & mapping_;

public:
  evaluate_expression_traversal(std::map<std::string, std::string> const & accessors, std::string & str, mapping_type const & mapping);
  void call_before_expansion(isaac::math_expression const & math_expression, std::size_t root_idx) const;
  void call_after_expansion(math_expression const & /*math_expression*/, std::size_t /*root_idx*/) const;
  void operator()(isaac::math_expression const & math_expression, std::size_t root_idx, leaf_t leaf) const;
};

std::string evaluate(leaf_t leaf, std::map<std::string, std::string> const & accessors,
                            isaac::math_expression const & math_expression, std::size_t root_idx, mapping_type const & mapping);

void evaluate(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                     math_expression const & expressions, mapping_type const & mappings);

/** @brief functor for fetching or writing-back the elements in a math_expression */
class process_traversal : public traversal_functor
{
public:
  process_traversal(std::map<std::string, std::string> const & accessors, kernel_generation_stream & stream,
                    mapping_type const & mapping, std::set<std::string> & already_processed);
  void operator()(math_expression const & math_expression, std::size_t root_idx, leaf_t leaf) const;
private:
  std::map<std::string, std::string> accessors_;
  kernel_generation_stream & stream_;
  mapping_type const & mapping_;
  std::set<std::string> & already_processed_;
};

void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
             isaac::math_expression const & math_expression, size_t root_idx, mapping_type const & mapping, std::set<std::string> & already_processed);

void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                    math_expression const & expressions, mapping_type const & mappings);


class math_expression_representation_functor : public traversal_functor{
private:
  static void append_id(char * & ptr, unsigned int val);
  void append(driver::Buffer const & h, numeric_type dtype, char prefix) const;
  void append(lhs_rhs_element const & lhs_rhs, bool is_assigned) const;
public:
  math_expression_representation_functor(symbolic_binder & binder, char *& ptr);
  void append(char*& p, const char * str) const;
  void operator()(isaac::math_expression const & math_expression, std::size_t root_idx, leaf_t leaf_t) const;
private:
  symbolic_binder & binder_;
  char *& ptr_;
};

}
#endif
