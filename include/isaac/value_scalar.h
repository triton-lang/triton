#ifndef ISAAC_VALUE_SCALAR_H
#define ISAAC_VALUE_SCALAR_H

#include "isaac/types.h"
#include "stdint.h"

namespace isaac
{

class scalar;
class array_expression;

union values_holder
{
//  int8_t bool8;
  int8_t int8;
  u_int8_t uint8;
  int16_t int16;
  u_int16_t uint16;
  int32_t int32;
  u_int32_t uint32;
  int64_t int64;
  u_int64_t uint64;
//  cl_half float16;
  cl_float float32;
  cl_double float64;
};

class value_scalar
{
  template<class T> void init(T const &);
  template<class T> T cast() const;
public:
#define ISAAC_INSTANTIATE(CLTYPE, ADTYPE) value_scalar(CLTYPE value, numeric_type dtype = ADTYPE);
//  ISAAC_INSTANTIATE(cl_bool, BOOL_TYPE)
  ISAAC_INSTANTIATE(int8_t, CHAR_TYPE)
  ISAAC_INSTANTIATE(uint8_t, UCHAR_TYPE)
  ISAAC_INSTANTIATE(int16_t, SHORT_TYPE)
  ISAAC_INSTANTIATE(u_int16_t, USHORT_TYPE)
  ISAAC_INSTANTIATE(int32_t, INT_TYPE)
  ISAAC_INSTANTIATE(u_int32_t, UINT_TYPE)
  ISAAC_INSTANTIATE(int64_t, LONG_TYPE)
  ISAAC_INSTANTIATE(u_int64_t, ULONG_TYPE)
//  ISAAC_INSTANTIATE(cl_half, HALF_TYPE)
  ISAAC_INSTANTIATE(cl_float, FLOAT_TYPE)
  ISAAC_INSTANTIATE(cl_double, DOUBLE_TYPE)
#undef ISAAC_INSTANTIATE
  value_scalar(values_holder value, numeric_type dtype);
  explicit value_scalar(scalar const &);
  explicit value_scalar(array_expression const &);
  explicit value_scalar(numeric_type dtype);

  values_holder values() const;
  numeric_type dtype() const;

#define INSTANTIATE(type) operator type() const;
  INSTANTIATE(bool)
  INSTANTIATE(int8_t)
  INSTANTIATE(uint8_t)
  INSTANTIATE(int16_t)
  INSTANTIATE(u_int16_t)
  INSTANTIATE(int32_t)
  INSTANTIATE(u_int32_t)
  INSTANTIATE(int64_t)
  INSTANTIATE(u_int64_t)
  INSTANTIATE(cl_float)
  INSTANTIATE(cl_double)
#undef INSTANTIATE
private:
  values_holder values_;
  numeric_type dtype_;
};

value_scalar int8(int8_t v);
value_scalar uint8(uint8_t v);
value_scalar int16(int16_t v);
value_scalar uint16(u_int16_t v);
value_scalar int32(int32_t v);
value_scalar uint32(u_int32_t v);
value_scalar int64(int64_t v);
value_scalar uint64(u_int64_t v);
value_scalar float32(cl_float v);
value_scalar float64(cl_double v);

template<class T>
T cast(isaac::value_scalar const &);

#define ISAAC_DECLARE_BINARY_OPERATOR(RET, OPNAME) \
RET OPNAME (value_scalar const &, int8_t  );\
RET OPNAME (value_scalar const &, uint8_t );\
RET OPNAME (value_scalar const &, int16_t );\
RET OPNAME (value_scalar const &, u_int16_t);\
RET OPNAME (value_scalar const &, int32_t   );\
RET OPNAME (value_scalar const &, u_int32_t  );\
RET OPNAME (value_scalar const &, int64_t  );\
RET OPNAME (value_scalar const &, u_int64_t );\
RET OPNAME (value_scalar const &, cl_float );\
RET OPNAME (value_scalar const &, cl_double);\
RET OPNAME (int8_t   , value_scalar const &);\
RET OPNAME (uint8_t  , value_scalar const &);\
RET OPNAME (int16_t  , value_scalar const &);\
RET OPNAME (u_int16_t , value_scalar const &);\
RET OPNAME (int32_t    , value_scalar const &);\
RET OPNAME (u_int32_t   , value_scalar const &);\
RET OPNAME (int64_t   , value_scalar const &);\
RET OPNAME (u_int64_t  , value_scalar const &);\
RET OPNAME (cl_float  , value_scalar const &);\
RET OPNAME (cl_double , value_scalar const &);\
RET OPNAME (value_scalar const &, value_scalar const &);

ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator +)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator -)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator *)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator /)

ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator >)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator >=)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator <)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator <=)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator ==)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, operator !=)

ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, max)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, min)
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, pow)

#undef ISAAC_DECLARE_BINARY_OPERATOR

//--------------------------------

//Unary operators
#define ISAAC_DECLARE_UNARY_OPERATOR(OPNAME) \
value_scalar OPNAME (value_scalar const & x);\

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

#undef ISAAC_DECLARE_UNARY_OPERATOR

std::ostream & operator<<(std::ostream & os, value_scalar const & s);

}

#endif
