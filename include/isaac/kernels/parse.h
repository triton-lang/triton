/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

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
  bool is_scalar_reduce_1d(expression_tree::node const & node);
  bool is_vector_reduce_1d(expression_tree::node const & node);
  bool is_assignment(op_element const & op);
  bool is_elementwise_operator(op_element const & op);
  bool is_elementwise_function(op_element const & op);
  bool is_cast(op_element const & op);
  bool bypass(op_element const & op);
}

class scalar;

/** @brief base functor class for traversing a expression_tree */
class traversal_functor
{
public:
  void call_before_expansion(expression_tree const &, std::size_t) const { }
  void call_after_expansion(expression_tree const &, std::size_t) const { }
};


/** @brief Recursively execute a functor on a expression_tree */
template<class Fun>
inline void traverse(isaac::expression_tree const & expression_tree, std::size_t root_idx, Fun const & fun, bool inspect)
{
  expression_tree::node const & root_node = expression_tree.tree()[root_idx];
  bool recurse = detail::is_node_leaf(root_node.op)?inspect:true;
  bool bypass = detail::bypass(root_node.op);

  if(!bypass)
    fun.call_before_expansion(expression_tree, root_idx);

  //Lhs:
  if (recurse)
  {
    if (root_node.lhs.subtype==COMPOSITE_OPERATOR_TYPE)
      traverse(expression_tree, root_node.lhs.node_index, fun, inspect);
    if (root_node.lhs.subtype != INVALID_SUBTYPE)
      fun(expression_tree, root_idx, LHS_NODE_TYPE);
  }

  //Self:
  if(!bypass)
    fun(expression_tree, root_idx, PARENT_NODE_TYPE);

  //Rhs:
  if (recurse && root_node.rhs.subtype!=INVALID_SUBTYPE)
  {
    if (root_node.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
      traverse(expression_tree, root_node.rhs.node_index, fun, inspect);
    if (root_node.rhs.subtype != INVALID_SUBTYPE)
      fun(expression_tree, root_idx, RHS_NODE_TYPE);
  }

  if(!bypass)
    fun.call_after_expansion(expression_tree, root_idx);
}

class filter_fun : public traversal_functor
{
public:
  typedef bool (*pred_t)(expression_tree::node const & node);
  filter_fun(pred_t pred, std::vector<size_t> & out);
  void operator()(isaac::expression_tree const & expression_tree, size_t root_idx, leaf_t) const;
private:
  pred_t pred_;
  std::vector<size_t> & out_;
};

class filter_elements_fun : public traversal_functor
{
public:
  filter_elements_fun(node_type subtype, std::vector<tree_node> & out);
  void operator()(isaac::expression_tree const & expression_tree, size_t root_idx, leaf_t) const;
private:
  node_type subtype_;
  std::vector<tree_node> & out_;
};

std::vector<size_t> filter_nodes(bool (*pred)(expression_tree::node const & node),
                                        isaac::expression_tree const & expression_tree,
                                        size_t root,
                                        bool inspect);

std::vector<tree_node> filter_elements(node_type subtype,
                                             isaac::expression_tree const & expression_tree);
const char * evaluate(operation_type type);

/** @brief functor for generating the expression string from a expression_tree */
class evaluate_expression_traversal: public traversal_functor
{
private:
  std::map<std::string, std::string> const & accessors_;
  std::string & str_;
  mapping_type const & mapping_;

public:
  evaluate_expression_traversal(std::map<std::string, std::string> const & accessors, std::string & str, mapping_type const & mapping);
  void call_before_expansion(isaac::expression_tree const & expression_tree, std::size_t root_idx) const;
  void call_after_expansion(expression_tree const & /*expression_tree*/, std::size_t /*root_idx*/) const;
  void operator()(isaac::expression_tree const & expression_tree, std::size_t root_idx, leaf_t leaf) const;
};

std::string evaluate(leaf_t leaf, std::map<std::string, std::string> const & accessors,
                            isaac::expression_tree const & expression_tree, std::size_t root_idx, mapping_type const & mapping);

void evaluate(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                     expression_tree const & expressions, mapping_type const & mappings);

/** @brief functor for fetching or writing-back the elements in a expression_tree */
class process_traversal : public traversal_functor
{
public:
  process_traversal(std::map<std::string, std::string> const & accessors, kernel_generation_stream & stream,
                    mapping_type const & mapping, std::set<std::string> & already_processed);
  void operator()(expression_tree const & expression_tree, std::size_t root_idx, leaf_t leaf) const;
private:
  std::map<std::string, std::string> accessors_;
  kernel_generation_stream & stream_;
  mapping_type const & mapping_;
  std::set<std::string> & already_processed_;
};

void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
             isaac::expression_tree const & expression_tree, size_t root_idx, mapping_type const & mapping, std::set<std::string> & already_processed);

void process(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors,
                    expression_tree const & expressions, mapping_type const & mappings);


class expression_tree_representation_functor : public traversal_functor{
private:
  static void append_id(char * & ptr, unsigned int val);
  void append(driver::Buffer const & h, numeric_type dtype, char prefix) const;
  void append(tree_node const & lhs_rhs, bool is_assigned) const;
public:
  expression_tree_representation_functor(symbolic_binder & binder, char *& ptr);
  void append(char*& p, const char * str) const;
  void operator()(isaac::expression_tree const & expression_tree, std::size_t root_idx, leaf_t leaf_t) const;
private:
  symbolic_binder & binder_;
  char *& ptr_;
};

}
#endif
