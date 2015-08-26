#include <cassert>
#include <vector>
#include "isaac/array.h"
#include "isaac/value_scalar.h"
#include "isaac/symbolic/expression.h"
#include "isaac/symbolic/preset.h"

namespace isaac
{

void fill(lhs_rhs_element &x, invalid_node)
{
  x.type_family = INVALID_TYPE_FAMILY;
  x.subtype = INVALID_SUBTYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
}

void fill(lhs_rhs_element & x, std::size_t node_index)
{
  x.type_family = COMPOSITE_OPERATOR_FAMILY;
  x.subtype = INVALID_SUBTYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
  x.node_index = node_index;
}

void fill(lhs_rhs_element & x, array const & a)
{
  x.type_family = ARRAY_TYPE_FAMILY;
  x.subtype = DENSE_ARRAY_TYPE;
  x.dtype = a.dtype();
  x.array = (array*)&a;
}

void fill(lhs_rhs_element & x, value_scalar const & v)
{
  x.type_family = VALUE_TYPE_FAMILY;
  x.dtype = v.dtype();
  x.subtype = VALUE_SCALAR_TYPE;
  x.vscalar = v.values();
}

void fill(lhs_rhs_element & x, repeat_infos const & r)
{
  x.type_family = INFOS_TYPE_FAMILY;
  x.subtype = REPEAT_INFOS_TYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
  x.tuple = r;
}

lhs_rhs_element::lhs_rhs_element(){}

//
op_element::op_element() {}
op_element::op_element(operation_node_type_family const & _type_family, operation_node_type const & _type) : type_family(_type_family), type(_type){}

//
template<class LT, class RT>
array_expression::array_expression(LT const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape) :
  tree_(1), root_(0), context_(context), dtype_(dtype), shape_(shape)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}

template<class RT>
array_expression::array_expression(array_expression const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape) :
 tree_(lhs.tree_.size() + 1), root_(tree_.size()-1), context_(context), dtype_(dtype), shape_(shape)
{
  std::copy(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  fill(tree_[root_].lhs, lhs.root_);
  tree_[root_].op = op;
  fill(tree_[root_].rhs, rhs);
}

template<class LT>
array_expression::array_expression(LT const & lhs, array_expression const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape) :
  tree_(rhs.tree_.size() + 1), root_(tree_.size() - 1), context_(context), dtype_(dtype), shape_(shape)
{
  std::copy(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin());
  fill(tree_[root_].lhs, lhs);
  tree_[root_].op = op;
  fill(tree_[root_].rhs, rhs.root_);
}

array_expression::array_expression(array_expression const & lhs, array_expression const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape):
  tree_(lhs.tree_.size() + rhs.tree_.size() + 1), root_(tree_.size()-1), context_(context), dtype_(dtype), shape_(shape)
{  
  std::size_t lsize = lhs.tree_.size();
  std::copy(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  std::copy(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin() + lsize);
  fill(tree_[root_].lhs, lhs.root_);
  tree_[root_].op = op;
  fill(tree_[root_].rhs, lsize + rhs.root_);
  for(container_type::iterator it = tree_.begin() + lsize ; it != tree_.end() - 1 ; ++it){
    if(it->lhs.type_family==COMPOSITE_OPERATOR_FAMILY) it->lhs.node_index+=lsize;
    if(it->rhs.type_family==COMPOSITE_OPERATOR_FAMILY) it->rhs.node_index+=lsize;
  }
  root_ = tree_.size() - 1;
}

template array_expression::array_expression(array_expression const &, value_scalar const &, op_element const &,  driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array_expression const &, invalid_node const &, op_element const &,  driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array_expression const &, array const &,        op_element const &,  driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array_expression const &, repeat_infos const &, op_element const &,  driver::Context const &, numeric_type const &, size4 const &);

template array_expression::array_expression(value_scalar const &, array_expression const &, op_element const &,  driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(invalid_node const &, array_expression const &, op_element const &,  driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array const &, array_expression const &, op_element const &,         driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(repeat_infos const &, array_expression const &, op_element const &,  driver::Context const &, numeric_type const &, size4 const &);

template array_expression::array_expression(value_scalar const &, value_scalar const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(invalid_node const &, value_scalar const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array const &,        value_scalar const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(repeat_infos const &, value_scalar const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);

template array_expression::array_expression(value_scalar const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(invalid_node const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array const &,        invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(repeat_infos const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);

template array_expression::array_expression(value_scalar const &, array const &,        op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(invalid_node const &, array const &,        op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array const &,        array const &,        op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(repeat_infos const &, array const &,        op_element const &, driver::Context const &, numeric_type const &, size4 const &);

template array_expression::array_expression(value_scalar const &, repeat_infos const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(invalid_node const &, repeat_infos const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(array const &,        repeat_infos const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);
template array_expression::array_expression(repeat_infos const &, repeat_infos const &, op_element const &, driver::Context const &, numeric_type const &, size4 const &);



array_expression::container_type & array_expression::tree()
{ return tree_; }

array_expression::container_type const & array_expression::tree() const
{ return tree_; }

std::size_t array_expression::root() const
{ return root_; }

driver::Context const & array_expression::context() const
{ return context_; }

numeric_type const & array_expression::dtype() const
{ return dtype_; }

size4 array_expression::shape() const
{ return shape_; }

int_t array_expression::nshape() const
{ return int_t((shape_[0] > 1) + (shape_[1] > 1)); }

array_expression& array_expression::reshape(int_t size1, int_t size2)
{
  assert(size1*size2==prod(shape_));
  shape_ = size4(size1, size2);
  return *this;
}

array_expression array_expression::operator-()
{ return array_expression(*this,  invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

array_expression array_expression::operator!()
{ return array_expression(*this, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_NEGATE_TYPE), context_, INT_TYPE, shape_); }

//
std::shared_ptr<array_expression> expressions_tuple::create(array_expression const & s)
{
  return std::shared_ptr<array_expression>(new array_expression(static_cast<array_expression const &>(s)));
}

expressions_tuple::expressions_tuple(data_type const & data, order_type order) : data_(data), order_(order)
{ }

expressions_tuple::expressions_tuple(array_expression const & s0) : order_(INDEPENDENT)
{
  data_.push_back(create(s0));
}

expressions_tuple::expressions_tuple(order_type order, array_expression const & s0, array_expression const & s1) : order_(order)
{
  data_.push_back(create(s0));
  data_.push_back(create(s1));
}

expressions_tuple::data_type const & expressions_tuple::data() const
{ return data_; }

driver::Context const & expressions_tuple::context() const
{ return data_.front()->context(); }

expressions_tuple::order_type expressions_tuple::order() const
{ return order_; }

array_expression::node const & lhs_most(array_expression::container_type const & array, array_expression::node const & init)
{
  array_expression::node const * current = &init;
  while (current->lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
    current = &array[current->lhs.node_index];
  return *current;
}

array_expression::node const & lhs_most(array_expression::container_type const & array, size_t root)
{ return lhs_most(array, array[root]); }


}
