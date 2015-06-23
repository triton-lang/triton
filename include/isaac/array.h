#ifndef ISAAC_ARRAY_H_
#define ISAAC_ARRAY_H_

#include <iostream>
#include "isaac/types.h"
#include "isaac/driver/backend.h"
#include "isaac/symbolic/expression.h"


namespace isaac
{

class scalar;

class array: public array_base
{
public:
  //1D Constructors
  array(int_t size1, numeric_type dtype, driver::Context context = driver::queues.default_context());
  array(int_t size1, numeric_type dtype, driver::Buffer data);
  template<typename DT>
  array(std::vector<DT> const & data, driver::Context context = driver::queues.default_context());
  array(array & v, slice const & s1);

  //2D Constructors
  array(int_t size1, int_t size2, numeric_type dtype, driver::Context context = driver::queues.default_context());
//  array(int_t size1, int_t size2, numeric_type dtype, cl_mem data);
  template<typename DT>
  array(int_t size1, int_t size2, std::vector<DT> const & data, driver::Context context = driver::queues.default_context());
  array(array & M, slice const & s1, slice const & s2);

  //3D Constructors
  array(int_t size1, int_t size2, int_t size3, numeric_type dtype, driver::Context context = driver::queues.default_context());

  //General constructor
  array(numeric_type dtype, driver::Buffer data, slice const & s1, slice const & s2, int_t ld, driver::Context context = driver::queues.default_context());

  array(array_expression const & proxy);
  array(array const &);
  template<class T>
  array(controller<T> const &);
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
  array& operator=(array_expression const &);
  template<class T>
  array& operator=(controller<T> const &);
  template<class T>
  array & operator=(std::vector<T> const & rhs);

  array_expression operator-();
  array_expression operator!();

  array& operator+=(value_scalar const &);
  array& operator+=(array const &);
  array& operator+=(array_expression const &);
  array& operator-=(value_scalar const &);
  array& operator-=(array const &);
  array& operator-=(array_expression const &);
  array& operator*=(value_scalar const &);
  array& operator*=(array const &);
  array& operator*=(array_expression const &);
  array& operator/=(value_scalar const &);
  array& operator/=(array const &);
  array& operator/=(array_expression const &);

  //Indexing operators
  const scalar operator[](int_t) const;
  scalar operator[](int_t);
  array operator[](slice const &);
  array operator()(slice const &, slice const &);

  array_expression T() const;
protected:
  numeric_type dtype_;

  size4 shape_;
  size4 start_;
  size4 stride_;
  int_t ld_;

  driver::Context context_;
  driver::Buffer data_;
};

class scalar : public array
{
  friend value_scalar::value_scalar(const scalar &);
  friend value_scalar::value_scalar(const array_expression &);
private:
  void inject(values_holder&) const;
  template<class T> T cast() const;
public:
  explicit scalar(numeric_type dtype, driver::Buffer const & data, int_t offset, driver::Context context = driver::queues.default_context());
  explicit scalar(value_scalar value, driver::Context context = driver::queues.default_context());
  explicit scalar(numeric_type dtype, driver::Context context = driver::queues.default_context());
  scalar(array_expression const & proxy);
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
void copy(void const * data, array & gx, driver::CommandQueue & queue, bool blocking = true);
void copy(array const & gx, void* data, driver::CommandQueue & queue, bool blocking = true);
void copy(void const *data, array &gx, bool blocking = true);
void copy(array const & gx, void* data, bool blocking = true);
template<class T> void copy(std::vector<T> const & cA, array& gA, driver::CommandQueue & queue, bool blocking = true);
template<class T> void copy(array const & gA, std::vector<T> & cA, driver::CommandQueue & queue, bool blocking = true);
template<class T> void copy(std::vector<T> const & cA, array & gA, bool blocking = true);
template<class T> void copy(array const & gA, std::vector<T> & cA, bool blocking = true);

//Operators
//Binary operators

#define ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(OPNAME) \
array_expression OPNAME (array_expression const & x, array_expression const & y);\
array_expression OPNAME (array const & x, array_expression const & y);\
array_expression OPNAME (array_expression const & x, array const & y);\
array_expression OPNAME (array const & x, array const & y);\
array_expression OPNAME (array_expression const & x, value_scalar const & y);\
array_expression OPNAME (array const & x, value_scalar const & y);\
array_expression OPNAME (value_scalar const & y, array_expression const & x);\
array_expression OPNAME (value_scalar const & y, array const & x);

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

namespace detail
{
  ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR(assign)
}

#undef ISAAC_DECLARE_ELEMENT_BINARY_OPERATOR

//--------------------------------

//Unary operators
#define ISAAC_DECLARE_UNARY_OPERATOR(OPNAME) \
  array_expression OPNAME (array const & x);\
  array_expression OPNAME (array_expression const & x);

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

array_expression cast(array const &, numeric_type dtype);
array_expression cast(array_expression const &, numeric_type dtype);

array_expression norm(array const &, unsigned int order = 2);
array_expression norm(array_expression const &, unsigned int order = 2);

#undef ISAAC_DECLARE_UNARY_OPERATOR

array_expression repmat(array const &, int_t const & rep1, int_t const & rep2);

#define ISAAC_DECLARE_REDUCTION(OPNAME) \
array_expression OPNAME(array const & M, int_t axis = -1);\
array_expression OPNAME(array_expression const & M, int_t axis = -1);

ISAAC_DECLARE_REDUCTION(sum)
ISAAC_DECLARE_REDUCTION(argmax)
ISAAC_DECLARE_REDUCTION(max)
ISAAC_DECLARE_REDUCTION(min)
ISAAC_DECLARE_REDUCTION(argmin)

array_expression eye(std::size_t, std::size_t, isaac::numeric_type, driver::Context context = driver::queues.default_context());
array_expression zeros(std::size_t M, std::size_t N, numeric_type dtype, driver::Context context = driver::queues.default_context());
array_expression reshape(array const &, int_t, int_t);

//
std::ostream& operator<<(std::ostream &, array const &);
std::ostream& operator<<(std::ostream & os, scalar const & s);

}
#endif
