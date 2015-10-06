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

class ISAACAPI array
{
//protected:
//  array(numeric_type dtype, driver::Buffer data, slice const & s1, slice const & s2, int_t ld);
public:
  //1D Constructors
  explicit array(int_t size1, numeric_type dtype = FLOAT_TYPE, driver::Context const & context = driver::backend::contexts::get_default());
  array(int_t size1, numeric_type dtype, driver::Buffer data, int_t start, int_t inc);

  template<typename DT>
  array(std::vector<DT> const & data, driver::Context const & context = driver::backend::contexts::get_default());
  array(array & v, slice const & s1);

  //2D Constructors
  array(int_t size1, int_t size2, numeric_type dtype = FLOAT_TYPE, driver::Context const & context = driver::backend::contexts::get_default());
  array(int_t size1, int_t size2, numeric_type dtype, driver::Buffer data, int_t start, int_t ld);
  template<typename DT>
  array(int_t size1, int_t size2, std::vector<DT> const & data, driver::Context const & context = driver::backend::contexts::get_default());
  array(array & M, slice const & s1, slice const & s2);

  //3D Constructors
  array(int_t size1, int_t size2, int_t size3, numeric_type dtype = FLOAT_TYPE, driver::Context const & context = driver::backend::contexts::get_default());

  //General constructor
  array(math_expression const & proxy);
  array(execution_handler const &);
  
  //Copy Constructor
  array(array const &);

  //Getters
  numeric_type dtype() const;
  size4 const & shape() const;
  int_t nshape() const;
  size4 const & start() const;
  size4 const & stride() const;
  int_t const & ld() const;
  driver::Context const & context() const;
  driver::Buffer const & data() const;
  driver::Buffer & data();
  int_t dsize() const;

  //Setters
  array& resize(int_t size1, int_t size2=1);

  //Numeric operators
  array& operator=(array const &);
  array& operator=(math_expression const &);
  array& operator=(execution_handler const &);
  template<class T>
  array & operator=(std::vector<T> const & rhs);
  array & operator=(value_scalar const & rhs);

  math_expression operator-();
  math_expression operator!();

  array& operator+=(value_scalar const &);
  array& operator+=(array const &);
  array& operator+=(math_expression const &);
  array& operator-=(value_scalar const &);
  array& operator-=(array const &);
  array& operator-=(math_expression const &);
  array& operator*=(value_scalar const &);
  array& operator*=(array const &);
  array& operator*=(math_expression const &);
  array& operator/=(value_scalar const &);
  array& operator/=(array const &);
  array& operator/=(math_expression const &);

  //Indexing (1D)
  math_expression operator[](for_idx_t idx) const;
  const scalar operator[](int_t) const;
  scalar operator[](int_t);
  view operator[](slice const &);

  //Indexing (2D)
  view operator()(slice const &, slice const &);
  view operator()(int_t, slice const &);
  view operator()(slice const &, int_t);


protected:
  numeric_type dtype_;

  size4 shape_;
  size4 start_;
  size4 stride_;
  int_t ld_;

  driver::Context context_;
  driver::Buffer data_;

public:
  math_expression T;
};

class ISAACAPI view : public array
{
public:
  view(array& data, slice const & s1);
  view(array& data, slice const & s1, slice const & s2);
};

class ISAACAPI scalar : public array
{
  friend value_scalar::value_scalar(const scalar &);
  friend value_scalar::value_scalar(const math_expression &);
private:
  void inject(values_holder&) const;
  template<class T> T cast() const;
public:
  explicit scalar(numeric_type dtype, const driver::Buffer &data, int_t offset);
  explicit scalar(value_scalar value, driver::Context const & context = driver::backend::contexts::get_default());
  explicit scalar(numeric_type dtype, driver::Context const & context = driver::backend::contexts::get_default());
  scalar(math_expression const & proxy);
  scalar& operator=(value_scalar const &);
//  scalar& operator=(scalar const & s);
  using array::operator =;

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
ISAACAPI void copy(void const * data, array & gx, driver::CommandQueue & queue, bool blocking = true);
ISAACAPI void copy(array const & gx, void* data, driver::CommandQueue & queue, bool blocking = true);
ISAACAPI void copy(void const *data, array &gx, bool blocking = true);
ISAACAPI void copy(array const & gx, void* data, bool blocking = true);
template<class T> ISAACAPI void copy(std::vector<T> const & cA, array& gA, driver::CommandQueue & queue, bool blocking = true);
template<class T> ISAACAPI void copy(array const & gA, std::vector<T> & cA, driver::CommandQueue & queue, bool blocking = true);
template<class T> ISAACAPI void copy(std::vector<T> const & cA, array & gA, bool blocking = true);
template<class T> ISAACAPI void copy(array const & gA, std::vector<T> & cA, bool blocking = true);

//Operators
//Binary operators

#define ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(OPNAME) \
ISAACAPI math_expression OPNAME (array const & x, math_expression const & y);\
ISAACAPI math_expression OPNAME (array const & x, value_scalar const & y);\
ISAACAPI math_expression OPNAME (array const & x, for_idx_t const & y);\
ISAACAPI math_expression OPNAME (array const & x, array const & y);\
\
ISAACAPI math_expression OPNAME (math_expression const & x, math_expression const & y);\
ISAACAPI math_expression OPNAME (math_expression const & x, value_scalar const & y);\
ISAACAPI math_expression OPNAME (math_expression const & x, for_idx_t const & y);\
ISAACAPI math_expression OPNAME (math_expression const & x, array const & y);\
\
ISAACAPI math_expression OPNAME (value_scalar const & y, math_expression const & x);\
ISAACAPI math_expression OPNAME (value_scalar const & y, for_idx_t const & x);\
ISAACAPI math_expression OPNAME (value_scalar const & y, array const & x);\
\
ISAACAPI math_expression OPNAME (for_idx_t const & y, math_expression const & x);\
ISAACAPI math_expression OPNAME (for_idx_t const & y, for_idx_t const & x);\
ISAACAPI math_expression OPNAME (for_idx_t const & y, value_scalar const & x);\
ISAACAPI math_expression OPNAME (for_idx_t const & y, array const & x);

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
  math_expression rot(LTYPE const & x, RTYPE const & y, CTYPE const & c, STYPE const & s);

ISAAC_DECLARE_ROT(array, array, scalar, scalar)
ISAAC_DECLARE_ROT(math_expression, array, scalar, scalar)
ISAAC_DECLARE_ROT(array, math_expression, scalar, scalar)
ISAAC_DECLARE_ROT(math_expression, math_expression, scalar, scalar)

ISAAC_DECLARE_ROT(array, array, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(math_expression, array, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(array, math_expression, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(math_expression, math_expression, value_scalar, value_scalar)

ISAAC_DECLARE_ROT(array, array, math_expression, math_expression)
ISAAC_DECLARE_ROT(math_expression, array, math_expression, math_expression)
ISAAC_DECLARE_ROT(array, math_expression, math_expression, math_expression)
ISAAC_DECLARE_ROT(math_expression, math_expression, math_expression, math_expression)
//--------------------------------


//Unary operators
#define ISAAC_DECLARE_UNARY_OPERATOR(OPNAME) \
  ISAACAPI math_expression OPNAME (array const & x);\
  ISAACAPI math_expression OPNAME (math_expression const & x);

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

ISAACAPI math_expression cast(array const &, numeric_type dtype);
ISAACAPI math_expression cast(math_expression const &, numeric_type dtype);

ISAACAPI math_expression norm(array const &, unsigned int order = 2);
ISAACAPI math_expression norm(math_expression const &, unsigned int order = 2);

#undef ISAAC_DECLARE_UNARY_OPERATOR

ISAACAPI math_expression repmat(array const &, int_t const & rep1, int_t const & rep2);

//Matrix reduction

#define ISAAC_DECLARE_DOT(OPNAME) \
ISAACAPI math_expression OPNAME(array const & M, int_t axis = -1);\
ISAACAPI math_expression OPNAME(math_expression const & M, int_t axis = -1);

ISAAC_DECLARE_DOT(sum)
ISAAC_DECLARE_DOT(argmax)
ISAAC_DECLARE_DOT((max))
ISAAC_DECLARE_DOT((min))
ISAAC_DECLARE_DOT(argmin)

//Fusion
ISAACAPI math_expression fuse(math_expression const & x, math_expression const & y);

//For
ISAACAPI math_expression sfor(math_expression const & start, math_expression const & end, math_expression const & inc, math_expression const & expression);
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
ISAACAPI math_expression eye(int_t, int_t, isaac::numeric_type, driver::Context const & context = driver::backend::contexts::get_default());
ISAACAPI math_expression zeros(int_t M, int_t N, numeric_type dtype, driver::Context const & context = driver::backend::contexts::get_default());

//Reshape
ISAACAPI math_expression reshape(array const &, int_t, int_t);

//diag
ISAACAPI math_expression diag(array const &, int offset = 0);
ISAACAPI math_expression diag(math_expression const &, int offset = 0);

//Row
ISAACAPI math_expression row(array const &, value_scalar const &);
ISAACAPI math_expression row(array const &, for_idx_t const &);
ISAACAPI math_expression row(array const &, math_expression const &);

ISAACAPI math_expression row(math_expression const &, value_scalar const &);
ISAACAPI math_expression row(math_expression const &, for_idx_t const &);
ISAACAPI math_expression row(math_expression const &, math_expression const &);

//col
ISAACAPI math_expression col(array const &, value_scalar const &);
ISAACAPI math_expression col(array const &, for_idx_t const &);
ISAACAPI math_expression col(array const &, math_expression const &);

ISAACAPI math_expression col(math_expression const &, value_scalar const &);
ISAACAPI math_expression col(math_expression const &, for_idx_t const &);
ISAACAPI math_expression col(math_expression const &, math_expression const &);


//
ISAACAPI std::ostream& operator<<(std::ostream &, array const &);

}
#endif
