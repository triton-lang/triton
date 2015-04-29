#ifndef ISAAC_VALUE_SCALAR_H
#define ISAAC_VALUE_SCALAR_H

#include "isaac/types.h"
#include <CL/cl.hpp>

namespace isaac
{

class scalar;
class array_expression;

union values_holder
{
//  cl_char bool8;
  cl_char int8;
  cl_uchar uint8;
  cl_short int16;
  cl_ushort uint16;
  cl_int int32;
  cl_uint uint32;
  cl_long int64;
  cl_ulong uint64;
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
  ISAAC_INSTANTIATE(cl_char, CHAR_TYPE)
  ISAAC_INSTANTIATE(cl_uchar, UCHAR_TYPE)
  ISAAC_INSTANTIATE(cl_short, SHORT_TYPE)
  ISAAC_INSTANTIATE(cl_ushort, USHORT_TYPE)
  ISAAC_INSTANTIATE(cl_int, INT_TYPE)
  ISAAC_INSTANTIATE(cl_uint, UINT_TYPE)
  ISAAC_INSTANTIATE(cl_long, LONG_TYPE)
  ISAAC_INSTANTIATE(cl_ulong, ULONG_TYPE)
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
private:
  values_holder values_;
  numeric_type dtype_;
};

value_scalar int8(cl_char v);
value_scalar uint8(cl_uchar v);
value_scalar int16(cl_short v);
value_scalar uint16(cl_ushort v);
value_scalar int32(cl_int v);
value_scalar uint32(cl_uint v);
value_scalar int64(cl_long v);
value_scalar uint64(cl_ulong v);
value_scalar float32(cl_float v);
value_scalar float64(cl_double v);

template<class T>
T cast(isaac::value_scalar const &);

#define ISAAC_DECLARE_BINARY_OPERATOR(RET, OPNAME) \
RET OPNAME (value_scalar const &, cl_char  );\
RET OPNAME (value_scalar const &, cl_uchar );\
RET OPNAME (value_scalar const &, cl_short );\
RET OPNAME (value_scalar const &, cl_ushort);\
RET OPNAME (value_scalar const &, cl_int   );\
RET OPNAME (value_scalar const &, cl_uint  );\
RET OPNAME (value_scalar const &, cl_long  );\
RET OPNAME (value_scalar const &, cl_ulong );\
RET OPNAME (value_scalar const &, cl_float );\
RET OPNAME (value_scalar const &, cl_double);\
RET OPNAME (cl_char   , value_scalar const &);\
RET OPNAME (cl_uchar  , value_scalar const &);\
RET OPNAME (cl_short  , value_scalar const &);\
RET OPNAME (cl_ushort , value_scalar const &);\
RET OPNAME (cl_int    , value_scalar const &);\
RET OPNAME (cl_uint   , value_scalar const &);\
RET OPNAME (cl_long   , value_scalar const &);\
RET OPNAME (cl_ulong  , value_scalar const &);\
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
