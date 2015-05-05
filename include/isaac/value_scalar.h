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
  int8_t int8;
  u_int8_t uint8;
  int16_t int16;
  u_int16_t uint16;
  int32_t int32;
  u_int32_t uint32;
  int64_t int64;
  u_int64_t uint64;
  float float32;
  double float64;
};

class value_scalar
{
  template<class T> void init(T const &);
  template<class T> T cast() const;
public:
#define ISAAC_INSTANTIATE(TYPE) value_scalar(TYPE value, numeric_type dtype = to_numeric_type<TYPE>::value);
  ISAAC_INSTANTIATE(char)
  ISAAC_INSTANTIATE(unsigned char)
  ISAAC_INSTANTIATE(short)
  ISAAC_INSTANTIATE(unsigned short)
  ISAAC_INSTANTIATE(int)
  ISAAC_INSTANTIATE(unsigned int)
  ISAAC_INSTANTIATE(long)
  ISAAC_INSTANTIATE(unsigned long)
  ISAAC_INSTANTIATE(float)
  ISAAC_INSTANTIATE(double)
#undef ISAAC_INSTANTIATE
  value_scalar(values_holder value, numeric_type dtype);
  explicit value_scalar(scalar const &);
  explicit value_scalar(array_expression const &);
  explicit value_scalar(numeric_type dtype);

  values_holder values() const;
  numeric_type dtype() const;

#define INSTANTIATE(type) operator type() const;
  INSTANTIATE(bool)
  INSTANTIATE(char)
  INSTANTIATE(unsigned char)
  INSTANTIATE(short)
  INSTANTIATE(unsigned short)
  INSTANTIATE(int)
  INSTANTIATE(unsigned int)
  INSTANTIATE(long)
  INSTANTIATE(unsigned long)
  INSTANTIATE(float)
  INSTANTIATE(double)
#undef INSTANTIATE
private:
  values_holder values_;
  numeric_type dtype_;
};

value_scalar int8(int8_t v);
value_scalar uint8(u_int8_t v);
value_scalar int16(int16_t v);
value_scalar uint16(u_int16_t v);
value_scalar int32(int32_t v);
value_scalar uint32(u_int32_t v);
value_scalar int64(int64_t v);
value_scalar uint64(u_int64_t v);
value_scalar float32(float v);
value_scalar float64(double v);

template<class T>
T cast(isaac::value_scalar const &);

#define ISAAC_DECLARE_BINARY_OPERATOR(RET, OPNAME) \
RET OPNAME (value_scalar const &, char  );\
RET OPNAME (value_scalar const &, unsigned char );\
RET OPNAME (value_scalar const &, short );\
RET OPNAME (value_scalar const &, unsigned short);\
RET OPNAME (value_scalar const &, int   );\
RET OPNAME (value_scalar const &, unsigned int  );\
RET OPNAME (value_scalar const &, long  );\
RET OPNAME (value_scalar const &, unsigned long );\
RET OPNAME (value_scalar const &, float );\
RET OPNAME (value_scalar const &, double);\
RET OPNAME (char   , value_scalar const &);\
RET OPNAME (unsigned char  , value_scalar const &);\
RET OPNAME (short  , value_scalar const &);\
RET OPNAME (unsigned short , value_scalar const &);\
RET OPNAME (int    , value_scalar const &);\
RET OPNAME (unsigned int   , value_scalar const &);\
RET OPNAME (long   , value_scalar const &);\
RET OPNAME (unsigned long  , value_scalar const &);\
RET OPNAME (float  , value_scalar const &);\
RET OPNAME (double , value_scalar const &);\
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
