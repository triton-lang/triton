#include <cassert>
#include <vector>
#include "atidlas/array.h"
#include "atidlas/value_scalar.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

lhs_rhs_element::lhs_rhs_element()
{
  type_family = INVALID_TYPE_FAMILY;
  subtype = INVALID_SUBTYPE;
  dtype = INVALID_NUMERIC_TYPE;
}

lhs_rhs_element::lhs_rhs_element(unsigned int _node_index)
{
  type_family = COMPOSITE_OPERATOR_FAMILY;
  subtype = INVALID_SUBTYPE;
  dtype = INVALID_NUMERIC_TYPE;
  node_index = _node_index;
}

lhs_rhs_element::lhs_rhs_element(atidlas::array const & x)
{
  type_family = ARRAY_TYPE_FAMILY;
  subtype = DENSE_ARRAY_TYPE;
  dtype = x.dtype();
  array = (atidlas::array*)&x;
}

lhs_rhs_element::lhs_rhs_element(atidlas::value_scalar const & x)
{
  type_family = VALUE_TYPE_FAMILY;
  dtype = x.dtype();
  subtype = VALUE_SCALAR_TYPE;
  vscalar = x.values();
}

lhs_rhs_element::lhs_rhs_element(atidlas::array_repeat_infos const & x)
{
  type_family = INFOS_TYPE_FAMILY;
  subtype = MATRIX_REPEAT_INFOS_TYPE;
  dtype = INVALID_NUMERIC_TYPE;
  repeat_infos = (atidlas::array_repeat_infos*)&x;
}

//
op_element::op_element(operation_node_type_family const & _type_family, operation_node_type const & _type) :
  type_family(_type_family), type(_type)
{ }

//
symbolic_expression_node::symbolic_expression_node(lhs_rhs_element const & _lhs, op_element const& _op, lhs_rhs_element const & _rhs) :
  lhs(_lhs), op(_op), rhs(_rhs)
{ }

//
symbolic_expression::symbolic_expression(lhs_rhs_element const & lhs, lhs_rhs_element const & rhs, op_element const & op, cl::Context const & context, numeric_type const & dtype) :
  array_(1, symbolic_expression_node(lhs, op, rhs)), root_(0), context_(context), dtype_(dtype)
{ }

symbolic_expression::symbolic_expression(symbolic_expression const & lhs, lhs_rhs_element const & rhs, op_element const & op) :
  context_(lhs.context_), dtype_(lhs.dtype_)
{
  array_.reserve(lhs.array_.size() + 1);
  array_.insert(array_.end(), lhs.array_.begin(), lhs.array_.end());
  array_.push_back(value_type(lhs_rhs_element(lhs.root_), op, rhs));
  root_ = array_.size() - 1;
}

symbolic_expression::symbolic_expression(lhs_rhs_element const & lhs, symbolic_expression const & rhs, op_element const & op) :
  context_(rhs.context_), dtype_(rhs.dtype_)
{
  array_.reserve(rhs.array_.size() + 1);
  array_.insert(array_.end(), rhs.array_.begin(), rhs.array_.end());
  array_.push_back(value_type(lhs, op, lhs_rhs_element(rhs.root_)));
  root_ = array_.size() - 1;
}

symbolic_expression::symbolic_expression(symbolic_expression const & lhs, symbolic_expression const & rhs, op_element const & op):
  context_(lhs.context_), dtype_(lhs.dtype_)
{
  std::size_t lsize = lhs.array_.size();
  std::size_t rsize = rhs.array_.size();
  array_.reserve(lsize + rsize + 1);
  array_.insert(array_.end(), lhs.array_.begin(), lhs.array_.end());
  array_.insert(array_.end(), rhs.array_.begin(), rhs.array_.end());
  array_.push_back(value_type(lhs_rhs_element(lhs.root_), op, lhs_rhs_element(lsize+rhs.root_)));
  for(container_type::iterator it = array_.begin() + lsize ; it != array_.end() - 1 ; ++it){
    if(it->lhs.type_family==COMPOSITE_OPERATOR_FAMILY) it->lhs.node_index+=lsize;
    if(it->rhs.type_family==COMPOSITE_OPERATOR_FAMILY) it->rhs.node_index+=lsize;
  }
  root_ = array_.size() - 1;
}

symbolic_expression::container_type & symbolic_expression::array()
{ return array_; }

symbolic_expression::container_type const & symbolic_expression::array() const
{ return array_; }

std::size_t symbolic_expression::root() const
{ return root_; }

cl::Context const & symbolic_expression::context() const
{ return context_; }

numeric_type const & symbolic_expression::dtype() const
{ return dtype_; }


//
array_expression::array_expression(lhs_rhs_element const & lhs, lhs_rhs_element const & rhs, op_element const & op, cl::Context const & ctx, numeric_type const & dtype, size4 shape):
      symbolic_expression(lhs, rhs, op, ctx, dtype), shape_(shape)
{ }

array_expression::array_expression(symbolic_expression const & lhs, lhs_rhs_element const & rhs, op_element const & op, size4 shape):
      symbolic_expression(lhs, rhs, op), shape_(shape)
{ }

array_expression::array_expression(lhs_rhs_element const & lhs, symbolic_expression const & rhs, op_element const & op, size4 shape):
  symbolic_expression(lhs, rhs, op), shape_(shape)
{ }

array_expression::array_expression(symbolic_expression const & lhs, symbolic_expression const & rhs, op_element const & op, size4 shape):
  symbolic_expression(lhs, rhs, op), shape_(shape)
{ }

size4 array_expression::shape() const
{ return shape_; }

int_t array_expression::nshape() const
{ return int_t((shape_._1 > 1) + (shape_._2 > 1)); }

array_expression& array_expression::reshape(int_t size1, int_t size2)
{
  assert(size1*size2==prod(shape_));
  shape_ = size4(size1, size2);
  return *this;
}


//
tools::shared_ptr<symbolic_expression> symbolic_expressions_container::create(symbolic_expression const & s)
{
  return tools::shared_ptr<symbolic_expression>(new array_expression(static_cast<array_expression const &>(s)));
}

symbolic_expressions_container::symbolic_expressions_container(data_type const & data, order_type order) : data_(data), order_(order)
{ }

symbolic_expressions_container::symbolic_expressions_container(symbolic_expression const & s0) : order_(INDEPENDENT)
{
  data_.push_back(create(s0));
}

symbolic_expressions_container::symbolic_expressions_container(order_type order, symbolic_expression const & s0, symbolic_expression const & s1) : order_(order)
{
  data_.push_back(create(s0));
  data_.push_back(create(s1));
}

symbolic_expressions_container::data_type const & symbolic_expressions_container::data() const
{ return data_; }

cl::Context const & symbolic_expressions_container::context() const
{ return data_.front()->context(); }

symbolic_expressions_container::order_type symbolic_expressions_container::order() const
{ return order_; }

symbolic_expression_node const & lhs_most(symbolic_expression::container_type const & array, symbolic_expression_node const & init)
{
  symbolic_expression_node const * current = &init;
  while (current->lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
    current = &array[current->lhs.node_index];
  return *current;
}

symbolic_expression_node const & lhs_most(symbolic_expression::container_type const & array, size_t root)
{ return lhs_most(array, array[root]); }


}
