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
  array_base& operator=(math_expression const &);
  array_base& operator=(execution_handler const &);
  template<class T>
  array_base & operator=(std::vector<T> const & rhs);
  array_base & operator=(value_scalar const & rhs);

  math_expression operator-();
  math_expression operator!();

  array_base& operator+=(value_scalar const &);
  array_base& operator+=(array_base const &);
  array_base& operator+=(math_expression const &);
  array_base& operator-=(value_scalar const &);
  array_base& operator-=(array_base const &);
  array_base& operator-=(math_expression const &);
  array_base& operator*=(value_scalar const &);
  array_base& operator*=(array_base const &);
  array_base& operator*=(math_expression const &);
  array_base& operator/=(value_scalar const &);
  array_base& operator/=(array_base const &);
  array_base& operator/=(math_expression const &);

  //Indexing (1D)
  math_expression operator[](for_idx_t idx) const;
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
  math_expression T;
};

class ISAACAPI array : public array_base
{
public:
  using array_base::array_base;
  //Copy Constructor
  array(array_base const &);
  array(array const &);
  array(math_expression const & proxy);
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
ISAACAPI math_expression OPNAME (array_base const & x, math_expression const & y);\
ISAACAPI math_expression OPNAME (array_base const & x, value_scalar const & y);\
ISAACAPI math_expression OPNAME (array_base const & x, for_idx_t const & y);\
ISAACAPI math_expression OPNAME (array_base const & x, array_base const & y);\
\
ISAACAPI math_expression OPNAME (math_expression const & x, math_expression const & y);\
ISAACAPI math_expression OPNAME (math_expression const & x, value_scalar const & y);\
ISAACAPI math_expression OPNAME (math_expression const & x, for_idx_t const & y);\
ISAACAPI math_expression OPNAME (math_expression const & x, array_base const & y);\
\
ISAACAPI math_expression OPNAME (value_scalar const & y, math_expression const & x);\
ISAACAPI math_expression OPNAME (value_scalar const & y, for_idx_t const & x);\
ISAACAPI math_expression OPNAME (value_scalar const & y, array_base const & x);\
\
ISAACAPI math_expression OPNAME (for_idx_t const & y, math_expression const & x);\
ISAACAPI math_expression OPNAME (for_idx_t const & y, for_idx_t const & x);\
ISAACAPI math_expression OPNAME (for_idx_t const & y, value_scalar const & x);\
ISAACAPI math_expression OPNAME (for_idx_t const & y, array_base const & x);

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

ISAAC_DECLARE_ROT(array_base, array_base, scalar, scalar)
ISAAC_DECLARE_ROT(math_expression, array_base, scalar, scalar)
ISAAC_DECLARE_ROT(array_base, math_expression, scalar, scalar)
ISAAC_DECLARE_ROT(math_expression, math_expression, scalar, scalar)

ISAAC_DECLARE_ROT(array_base, array_base, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(math_expression, array_base, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(array_base, math_expression, value_scalar, value_scalar)
ISAAC_DECLARE_ROT(math_expression, math_expression, value_scalar, value_scalar)

ISAAC_DECLARE_ROT(array_base, array_base, math_expression, math_expression)
ISAAC_DECLARE_ROT(math_expression, array_base, math_expression, math_expression)
ISAAC_DECLARE_ROT(array_base, math_expression, math_expression, math_expression)
ISAAC_DECLARE_ROT(math_expression, math_expression, math_expression, math_expression)
//--------------------------------


//Unary operators
#define ISAAC_DECLARE_UNARY_OPERATOR(OPNAME) \
  ISAACAPI math_expression OPNAME (array_base const & x);\
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

ISAACAPI math_expression cast(array_base const &, numeric_type dtype);
ISAACAPI math_expression cast(math_expression const &, numeric_type dtype);

ISAACAPI math_expression norm(array_base const &, unsigned int order = 2);
ISAACAPI math_expression norm(math_expression const &, unsigned int order = 2);

#undef ISAAC_DECLARE_UNARY_OPERATOR

ISAACAPI math_expression repmat(array_base const &, int_t const & rep1, int_t const & rep2);

//Matrix reduction

#define ISAAC_DECLARE_DOT(OPNAME) \
ISAACAPI math_expression OPNAME(array_base const & M, int_t axis = -1);\
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

//Swap
ISAACAPI void swap(view x, view y);

//Reshape
ISAACAPI math_expression reshape(array_base const &, shape_t const &);
ISAACAPI math_expression ravel(array_base const &);

//diag
array diag(array_base & x, int offset = 0);

//Row
ISAACAPI math_expression row(array_base const &, value_scalar const &);
ISAACAPI math_expression row(array_base const &, for_idx_t const &);
ISAACAPI math_expression row(array_base const &, math_expression const &);

ISAACAPI math_expression row(math_expression const &, value_scalar const &);
ISAACAPI math_expression row(math_expression const &, for_idx_t const &);
ISAACAPI math_expression row(math_expression const &, math_expression const &);

//col
ISAACAPI math_expression col(array_base const &, value_scalar const &);
ISAACAPI math_expression col(array_base const &, for_idx_t const &);
ISAACAPI math_expression col(array_base const &, math_expression const &);

ISAACAPI math_expression col(math_expression const &, value_scalar const &);
ISAACAPI math_expression col(math_expression const &, for_idx_t const &);
ISAACAPI math_expression col(math_expression const &, math_expression const &);


//
ISAACAPI std::ostream& operator<<(std::ostream &, array_base const &);
ISAACAPI std::ostream& operator<<(std::ostream &, math_expression const &);

}
#endif
