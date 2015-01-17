#ifndef ATIDLAS_ARRAY_H_
#define ATIDLAS_ARRAY_H_

#include <iostream>
#include "atidlas/types.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/cl/queues.h"
#include "atidlas/symbolic/expression.h"


namespace atidlas
{

class scalar;

class array: public obj_base
{
public:
  //1D Constructors
  array(int_t size1, numeric_type dtype, cl::Context context = cl::default_context());
  template<typename T>
  array(std::vector<T> const & data, cl::Context context = cl::default_context());
  array(array & v, slice const & s1);

  //2D Constructors
  array(int_t size1, int_t size2, numeric_type dtype, cl::Context context = cl::default_context());
  template<typename T>
  array(int_t size1, int_t size2, std::vector<T> const & data, cl::Context context = cl::default_context());
  array(array & M, slice const & s1, slice const & s2);

  //General constructor
  array(numeric_type dtype, cl::Buffer data, slice const & s1, slice const & s2, cl::Context context = cl::default_context());
  explicit array(array_expression const & proxy);

  //Getters
  numeric_type dtype() const;
  size4 shape() const;
  int_t nshape() const;
  size4 start() const;
  size4 stride() const;
  int_t ld() const;
  cl::Context const & context() const;
  cl::Buffer const & data() const;
  int_t dsize() const;

  //Setters
  array& resize(int_t size1, int_t size2=1);
  array& reshape(int_t size1, int_t size2=1);

  //Numeric operators
  array& operator=(array const &);
  array& operator=(array_expression const &);
  template<class T> array & operator=(std::vector<T> const & rhs);

  array& operator-();
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
protected:
  numeric_type dtype_;

  size4 shape_;
  size4 start_;
  size4 stride_;
  int_t ld_;

  cl::Context context_;
  cl::Buffer data_;
};

class scalar : public array
{
private:
  template<class T> T cast() const;
public:
  explicit scalar(numeric_type dtype, cl::Buffer const & data, int_t offset, cl::Context context = cl::default_context());
  explicit scalar(value_scalar value, cl::Context context = cl::default_context());
  explicit scalar(numeric_type dtype, cl::Context context = cl::default_context());
  scalar(array_expression const & proxy);

  scalar& operator=(value_scalar const &);
  scalar& operator=(scalar const &);
  using array::operator=;

#define INSTANTIATE(type) operator type() const;
  INSTANTIATE(bool)
  INSTANTIATE(cl_char)
  INSTANTIATE(cl_uchar)
  INSTANTIATE(cl_short)
  INSTANTIATE(cl_ushort)
  INSTANTIATE(cl_int)
  INSTANTIATE(cl_uint)
  INSTANTIATE(cl_long)
  INSTANTIATE(cl_ulong)
  INSTANTIATE(cl_float)
  INSTANTIATE(cl_double)
#undef INSTANTIATE
};


atidlas::array_expression eye(std::size_t, std::size_t, atidlas::numeric_type, cl::Context ctx = cl::default_context());

array_expression zeros(std::size_t N, numeric_type dtype);

//copy

void copy(void const * data, array & gx, cl::CommandQueue & queue, bool blocking = true);
void copy(array const & gx, void* data, cl::CommandQueue & queue, bool blocking = true);
void copy(void const *data, array &gx, bool blocking = true);
void copy(array const & gx, void* data, bool blocking = true);
template<class T> void copy(std::vector<T> const & cA, array& gA, cl::CommandQueue & queue, bool blocking = true);
template<class T> void copy(array const & gA, std::vector<T> & cA, cl::CommandQueue & queue, bool blocking = true);
template<class T> void copy(std::vector<T> const & cA, array & gA, bool blocking = true);
template<class T> void copy(array const & gA, std::vector<T> & cA, bool blocking = true);

//Operators
//Binary operators

#define ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(OPNAME) \
array_expression OPNAME (array_expression const & x, array_expression const & y);\
array_expression OPNAME (array const & x, array_expression const & y);\
array_expression OPNAME (array_expression const & x, array const & y);\
array_expression OPNAME (array const & x, array const & y);\
array_expression OPNAME (array_expression const & x, value_scalar const & y);\
array_expression OPNAME (array const & x, value_scalar const & y);\
array_expression OPNAME (value_scalar const & y, array_expression const & x);\
array_expression OPNAME (value_scalar const & y, array const & x);

ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator +)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator -)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator *)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator /)

ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator >)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator >=)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator <)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator <=)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator ==)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(operator !=)

ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(max)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(min)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(pow)

ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(dot)
ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(outer)

namespace detail
{
  ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR(assign)
}

#undef ATIDLAS_DECLARE_ELEMENT_BINARY_OPERATOR

//--------------------------------

//Unary operators
#define ATIDLAS_DECLARE_UNARY_OPERATOR(OPNAME) \
  array_expression OPNAME (array const & x);\
  array_expression OPNAME (array_expression const & x);

ATIDLAS_DECLARE_UNARY_OPERATOR(abs)
ATIDLAS_DECLARE_UNARY_OPERATOR(acos)
ATIDLAS_DECLARE_UNARY_OPERATOR(asin)
ATIDLAS_DECLARE_UNARY_OPERATOR(atan)
ATIDLAS_DECLARE_UNARY_OPERATOR(ceil)
ATIDLAS_DECLARE_UNARY_OPERATOR(cos)
ATIDLAS_DECLARE_UNARY_OPERATOR(cosh)
ATIDLAS_DECLARE_UNARY_OPERATOR(exp)
ATIDLAS_DECLARE_UNARY_OPERATOR(floor)
ATIDLAS_DECLARE_UNARY_OPERATOR(log)
ATIDLAS_DECLARE_UNARY_OPERATOR(log10)
ATIDLAS_DECLARE_UNARY_OPERATOR(sin)
ATIDLAS_DECLARE_UNARY_OPERATOR(sinh)
ATIDLAS_DECLARE_UNARY_OPERATOR(sqrt)
ATIDLAS_DECLARE_UNARY_OPERATOR(tan)
ATIDLAS_DECLARE_UNARY_OPERATOR(tanh)
ATIDLAS_DECLARE_UNARY_OPERATOR(trans)

array_expression norm(array const &, unsigned int order = 2);
array_expression norm(array_expression const &, unsigned int order = 2);

#undef ATIDLAS_DECLARE_UNARY_OPERATOR

struct repeat_infos
{
    repeat_infos(size4 const & _sub, size4 const & _rep) : sub(_sub), rep(_rep){ }
    size4 sub;
    size4 rep;
};

array_expression repmat(array const &, int_t const & rep1, int_t const & rep2);

#define ATIDLAS_DECLARE_REDUCTION(OPNAME) \
array_expression OPNAME(array const & M, int_t axis = -1);\
array_expression OPNAME(array_expression const & M, int_t axis = -1);

ATIDLAS_DECLARE_REDUCTION(sum)
ATIDLAS_DECLARE_REDUCTION(argmax)
ATIDLAS_DECLARE_REDUCTION(max)
ATIDLAS_DECLARE_REDUCTION(min)
ATIDLAS_DECLARE_REDUCTION(argmin)

//
std::ostream& operator<<(std::ostream &, array const &);
std::ostream& operator<<(std::ostream &, array_expression const &);

}
#endif
