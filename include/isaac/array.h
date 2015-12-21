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
#ifndef ISAAC_ARRAY_H_
#define ISAAC_ARRAY_H_

#include <iostream>
#include "isaac/defines.h"
#include "isaac/driver/backend.h"
#include "isaac/symbolic/expression.h"
#include "isaac/types.h"


namespace isaac
{

class scalar;
class view;

class ISAACAPI array_base
{
  int_t dsize();
public:
  //1D Constructors
  explicit array_base(int_t size1, numeric_type dtype = FLOAT_TYPE, driver::Context const & context = driver::backend::contexts::get_default());
  array_base(int_t size1, numeric_type dtype, driver::Buffer data, int_t start, int_t inc);

  template<typename DT>
  array_base(std::vector<DT> const & data, driver::Context const & context = driver::backend::contexts::get_default());
  array_base(array_base & v, slice const & s1);

  //2D Constructors
  array_base(int_t size1, int_t size2, numeric_type dtype = FLOAT_TYPE, driver::Context const & context = driver::backend::contexts::get_default());
  array_base(int_t size1, int_t size2, numeric_type dtype, driver::Buffer data, int_t start, int_t ld);
  template<typename DT>
  array_base(int_t size1, int_t size2, std::vector<DT> const & data, driver::Context const & context = driver::backend::contexts::get_default());
  array_base(array_base & M, slice const & s1, slice const & s2);

  //3D Constructors
  array_base(int_t size1, int_t size2, int_t size3, numeric_type dtype = FLOAT_TYPE, driver::Context const & context = driver::backend::contexts::get_default());

  //General constructor
  array_base(numeric_type dtype, shape_t const & shape, driver::Context const & context);
  array_base(numeric_type dtype, shape_t const & shape, int_t start, shape_t const & stride, driver::Context const & context);
  explicit array_base(execution_handler const &);

  //Make the class virtual
  virtual ~array_base() = 0;

  //Getters
  numeric_type dtype() const;
  shape_t const & shape() const;
  int_t dim() const;
  int_t start() const;
  shape_t const & stride() const;
  driver::Context const & context() const;
  driver::Buffer const & data() const;
  driver::Buffer & data();

  //Setters
  array_base& resize(int_t size1, int_t size2=1);

  //Numeric operators
  array_base& operator=(array_base const &);
  array_base& operator=(expression_tree const &);
  array_base& operator=(execution_handler const &);
  template<class T>
  array_base & operator=(std::vector<T> const & rhs);
  array_base & operator=(value_scalar const & rhs);

  expression_tree operator-();
  expression_tree operator!();

  array_base& operator+=(value_scalar const &);
  array_base& operator+=(array_base const &);
  array_base& operator+=(expression_tree const &);
  array_base& operator-=(value_scalar const &);
  array_base& operator-=(array_base const &);
  array_base& operator-=(expression_tree const &);
  array_base& operator*=(value_scalar const &);
  array_base& operator*=(array_base const &);
  array_base& operator*=(expression_tree const &);
  array_base& operator/=(value_scalar const &);
  array_base& operator/=(array_base const &);
  array_base& operator/=(expression_tree const &);

  //Indexing (1D)
  expression_tree operator[](for_idx_t idx) const;
  const scalar operator[](int_t) const;
  scalar operator[](int_t);
  view operator[](slice const &);

  //Indexing (2D)
  view operator()(int_t, int_t);
  view operator()(slice const &, int_t);
  view operator()(int_t, slice const &);
  view operator()(slice const &, slice const &);


protected:
  numeric_type dtype_;

  shape_t shape_;
  int_t start_;
  shape_t stride_;

  driver::Context context_;
  driver::Buffer data_;

public:
  expression_tree T;
};

class ISAACAPI array : public array_base
{
public:
  using array_base::array_base;
  //Copy Constructor
  array(array_base const &);
  array(array const &);
  array(expression_tree const & proxy);
  using array_base::operator=;
};

class ISAACAPI view : public array_base
{
public:
  view(array & data);
  view(array_base& data, slice const & s1);
  view(array_base& data, slice const & s1, slice const & s2);
  view(int_t size1, numeric_type dtype, driver::Buffer data, int_t start, int_t inc);
  using array_base::operator=;
};

class ISAACAPI scalar : public array_base
{
  friend value_scalar::value_scalar(const scalar &);
  friend value_scalar::value_scalar(const expression_tree &);
private:
  void inject(values_holder&) const;
  template<class T> T cast() const;
public:
  explicit scalar(numeric_type dtype, const driver::Buffer &data, int_t offset);
  explicit scalar(value_scalar value, driver::Context const & context = driver::backend::contexts::get_default());
  explicit scalar(numeric_type dtype, driver::Context const & context = driver::backend::contexts::get_default());
  scalar(expression_tree const & proxy);
  scalar& operator=(value_scalar const &);
//  scalar& operator=(scalar const & s);
  using array_base::operator =;

#define INSTANTIATE(type) operator type() const;
  INSTANTIATE(char)
  INSTANTIATE(unsigned char)
  INSTANTIATE(short)
  INSTANTIATE(unsigned short)
  INSTANTIATE(int)
  INSTANTIATE(unsigned int)
  INSTANTIATE(long)
  INSTANTIATE(unsigned long)
  INSTANTIATE(long long)
  INSTANTIATE(unsigned long long)
  INSTANTIATE(float)
  INSTANTIATE(double)
#undef INSTANTIATE
};




//copy
ISAACAPI void copy(void const * data, array_base & gx, driver::CommandQueue & queue, bool blocking = true);
ISAACAPI void copy(array_base const & gx, void* data, driver::CommandQueue & queue, bool blocking = true);
ISAACAPI void copy(void const *data, array_base &gx, bool blocking = true);
ISAACAPI void copy(array_base const & gx, void* data, bool blocking = true);
template<class T> ISAACAPI void copy(std::vector<T> const & cA, array_base& gA, driver::CommandQueue & queue, bool blocking = true);
template<class T> ISAACAPI void copy(array_base const & gA, std::vector<T> & cA, driver::CommandQueue & queue, bool blocking = true);
template<class T> ISAACAPI void copy(std::vector<T> const & cA, array_base & gA, bool blocking = true);
template<class T> ISAACAPI void copy(array_base const & gA, std::vector<T> & cA, bool blocking = true);

//Operators
//Binary operators

#define ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(OPNAME) \
ISAACAPI expression_tree OPNAME (array_base const & x, expression_tree const & y);\
ISAACAPI expression_tree OPNAME (array_base const & x, value_scalar const & y);\
ISAACAPI expression_tree OPNAME (array_base const & x, for_idx_t const & y);\
ISAACAPI expression_tree OPNAME (array_base const & x, array_base const & y);\
\
ISAACAPI expression_tree OPNAME (expression_tree const & x, expression_tree const & y);\
ISAACAPI expression_tree OPNAME (expression_tree const & x, value_scalar const & y);\
ISAACAPI expression_tree OPNAME (expression_tree const & x, for_idx_t const & y);\
ISAACAPI expression_tree OPNAME (expression_tree const & x, array_base const & y);\
\
ISAACAPI expression_tree OPNAME (value_scalar const & y, expression_tree const & x);\
ISAACAPI expression_tree OPNAME (value_scalar const & y, for_idx_t const & x);\
ISAACAPI expression_tree OPNAME (value_scalar const & y, array_base const & x);\
\
ISAACAPI expression_tree OPNAME (for_idx_t const & y, expression_tree const & x);\
ISAACAPI expression_tree OPNAME (for_idx_t const & y, for_idx_t const & x);\
ISAACAPI expression_tree OPNAME (for_idx_t const & y, value_scalar const & x);\
ISAACAPI expression_tree OPNAME (for_idx_t const & y, array_base const & x);

ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator +)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator -)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator *)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator /)

ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator >)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator >=)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator <)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator <=)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator ==)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(operator !=)

ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(maximum)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(minimum)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(pow)

ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(dot)
ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(outer)

ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(assign)

#undef ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR

#define ISAAC_DECLARE_ROT(LTYPE, RTYPE, CTYPE, STYPE) \
  expression_tree rot(LTYPE const & x, RTYPE const & y, CTYPE const & c, STYPE const & s);

ISAAC_DECLARE_ROT(array_base, array_base, scalar, scalar)
ISAAC_DECLARE_ROT(expression_tree, array_base, scalar, scalar)
ISAAC_DECLARE_ROT(array_base, expression_tree, scalar, scalar)
ISAAC_DECLARE_ROT(expression_tree, expression_tree, scalar, scalar)

ISAAC_DECLARE_ROT(array_base, array_base, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(expression_tree, array_base, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(array_base, expression_tree, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(expression_tree, expression_tree, value_scalar, value_scalar)

ISAAC_DECLARE_ROT(array_base, array_base, expression_tree, expression_tree)
ISAAC_DECLARE_ROT(expression_tree, array_base, expression_tree, expression_tree)
ISAAC_DECLARE_ROT(array_base, expression_tree, expression_tree, expression_tree)
ISAAC_DECLARE_ROT(expression_tree, expression_tree, expression_tree, expression_tree)
//--------------------------------


//Unary operators
#define ISAAC_DECLARE_UNARY_OPERATOR(OPNAME) \
  ISAACAPI expression_tree OPNAME (array_base const & x);\
  ISAACAPI expression_tree OPNAME (expression_tree const & x);

ISAAC_DECLARE_UNARY_OPERATOR(abs)
ISAAC_DECLARE_UNARY_OPERATOR(acos)
ISAAC_DECLARE_UNARY_OPERATOR(asin)
ISAAC_DECLARE_UNARY_OPERATOR(atan)
ISAAC_DECLARE_UNARY_OPERATOR(ceil)
ISAAC_DECLARE_UNARY_OPERATOR(cos)
ISAAC_DECLARE_UNARY_OPERATOR(cosh)
ISAAC_DECLARE_UNARY_OPERATOR(exp)
ISAAC_DECLARE_UNARY_OPERATOR(floor)
ISAAC_DECLARE_UNARY_OPERATOR(log)
ISAAC_DECLARE_UNARY_OPERATOR(log10)
ISAAC_DECLARE_UNARY_OPERATOR(sin)
ISAAC_DECLARE_UNARY_OPERATOR(sinh)
ISAAC_DECLARE_UNARY_OPERATOR(sqrt)
ISAAC_DECLARE_UNARY_OPERATOR(tan)
ISAAC_DECLARE_UNARY_OPERATOR(tanh)
ISAAC_DECLARE_UNARY_OPERATOR(trans)

ISAACAPI expression_tree cast(array_base const &, numeric_type dtype);
ISAACAPI expression_tree cast(expression_tree const &, numeric_type dtype);

ISAACAPI expression_tree norm(array_base const &, unsigned int order = 2);
ISAACAPI expression_tree norm(expression_tree const &, unsigned int order = 2);

#undef ISAAC_DECLARE_UNARY_OPERATOR

ISAACAPI expression_tree repmat(array_base const &, int_t const & rep1, int_t const & rep2);

//Matrix reduction

#define ISAAC_DECLARE_DOT(OPNAME) \
ISAACAPI expression_tree OPNAME(array_base const & M, int_t axis = -1);\
ISAACAPI expression_tree OPNAME(expression_tree const & M, int_t axis = -1);

ISAAC_DECLARE_DOT(sum)
ISAAC_DECLARE_DOT(argmax)
ISAAC_DECLARE_DOT((max))
ISAAC_DECLARE_DOT((min))
ISAAC_DECLARE_DOT(argmin)

//Fusion
ISAACAPI expression_tree fuse(expression_tree const & x, expression_tree const & y);

//For
ISAACAPI expression_tree sfor(expression_tree const & start, expression_tree const & end, expression_tree const & inc, expression_tree const & expression);
static const for_idx_t _i0{0};
static const for_idx_t _i1{1};
static const for_idx_t _i2{2};
static const for_idx_t _i3{3};
static const for_idx_t _i4{4};
static const for_idx_t _i5{5};
static const for_idx_t _i6{6};
static const for_idx_t _i7{7};
static const for_idx_t _i8{8};
static const for_idx_t _i9{9};

//Initializers
ISAACAPI expression_tree eye(int_t, int_t, isaac::numeric_type, driver::Context const & context = driver::backend::contexts::get_default());
ISAACAPI expression_tree zeros(shape_t const & shape, numeric_type dtype, driver::Context const & context = driver::backend::contexts::get_default());

//Swap
ISAACAPI void swap(view x, view y);

//Reshape
ISAACAPI expression_tree reshape(array_base const &, shape_t const &);
ISAACAPI expression_tree ravel(array_base const &);

//diag
array diag(array_base & x, int offset = 0);

//Row
ISAACAPI expression_tree row(array_base const &, value_scalar const &);
ISAACAPI expression_tree row(array_base const &, for_idx_t const &);
ISAACAPI expression_tree row(array_base const &, expression_tree const &);

ISAACAPI expression_tree row(expression_tree const &, value_scalar const &);
ISAACAPI expression_tree row(expression_tree const &, for_idx_t const &);
ISAACAPI expression_tree row(expression_tree const &, expression_tree const &);

//col
ISAACAPI expression_tree col(array_base const &, value_scalar const &);
ISAACAPI expression_tree col(array_base const &, for_idx_t const &);
ISAACAPI expression_tree col(array_base const &, expression_tree const &);

ISAACAPI expression_tree col(expression_tree const &, value_scalar const &);
ISAACAPI expression_tree col(expression_tree const &, for_idx_t const &);
ISAACAPI expression_tree col(expression_tree const &, expression_tree const &);


//
ISAACAPI std::ostream& operator<<(std::ostream &, array_base const &);
ISAACAPI std::ostream& operator<<(std::ostream &, expression_tree const &);

}
#endif
