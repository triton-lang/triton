#ifndef ATIDLAS_VALUE_SCALAR_H
#define ATIDLAS_VALUE_SCALAR_H

#include "atidlas/types.h"
#include "atidlas/cl/cl.hpp"

namespace atidlas
{

class scalar;
class array_expression;

union values_holder
{
  cl_char int8;
  cl_uchar uint8;
  cl_short int16;
  cl_ushort uint16;
  cl_int int32;
  cl_uint uint32;
  cl_long int64;
  cl_ulong uint64;
//    cl_half float16;
  cl_float float32;
  cl_double float64;
};

class value_scalar
{
  void init(scalar const &);
  template<class T> T cast() const;
public:
  value_scalar(cl_char value);
  value_scalar(cl_uchar value);
  value_scalar(cl_short value);
  value_scalar(cl_ushort value);
  value_scalar(cl_int value);
  value_scalar(cl_uint value);
  value_scalar(cl_long value);
  value_scalar(cl_ulong value);
//  value_scalar(cl_half value);
  value_scalar(cl_float value);
  value_scalar(cl_double value);
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
T cast(atidlas::value_scalar const &);

#define ATIDLAS_DECLARE_BINARY_OPERATOR(RET, OPNAME) \
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

ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator +)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator -)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator *)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator /)

ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator >)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator >=)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator <)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator <=)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator ==)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, operator !=)

ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, max)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, min)
ATIDLAS_DECLARE_BINARY_OPERATOR(value_scalar, pow)

#undef ATIDLAS_DECLARE_BINARY_OPERATOR

//--------------------------------

//Unary operators
#define ATIDLAS_DECLARE_UNARY_OPERATOR(OPNAME) \
value_scalar OPNAME (value_scalar const & x);\

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

#undef ATIDLAS_DECLARE_UNARY_OPERATOR

std::ostream & operator<<(std::ostream & os, value_scalar const & s);

}

#endif
