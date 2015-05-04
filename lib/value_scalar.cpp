#include <cmath>
#include <iostream>
#include "isaac/array.h"
#include "isaac/value_scalar.h"
#include "isaac/exception/unknown_datatype.h"

namespace isaac
{

template<class T>
void value_scalar::init(T const & s)
{
  switch(dtype_)
  {
//    case BOOL_TYPE: values_.bool8 = s; break;
    case CHAR_TYPE: values_.int8 = s; break;
    case UCHAR_TYPE: values_.uint8 = s; break;
    case SHORT_TYPE: values_.int16 = s; break;
    case USHORT_TYPE: values_.uint16 = s; break;
    case INT_TYPE: values_.int32 = s; break;
    case UINT_TYPE: values_.uint32 = s; break;
    case LONG_TYPE: values_.int64 = s; break;
    case ULONG_TYPE: values_.uint64 = s; break;
//    case HALF_TYPE: values_.float16 = s; break;
    case FLOAT_TYPE: values_.float32 = s; break;
    case DOUBLE_TYPE: values_.float64 = s; break;
    default: throw unknown_datatype(dtype_);
  }
}

#define INSTANTIATE(CLTYPE, ADTYPE) value_scalar::value_scalar(CLTYPE value, numeric_type dtype) : dtype_(dtype) { init(value); }

INSTANTIATE(int8_t, CHAR_TYPE)
INSTANTIATE(u_int8_t, UCHAR_TYPE)
INSTANTIATE(int16_t, SHORT_TYPE)
INSTANTIATE(u_int16_t, USHORT_TYPE)
INSTANTIATE(int32_t, INT_TYPE)
INSTANTIATE(u_int32_t, UINT_TYPE)
INSTANTIATE(int64_t, LONG_TYPE)
INSTANTIATE(u_int64_t, ULONG_TYPE)
INSTANTIATE(cl_float, FLOAT_TYPE)
INSTANTIATE(cl_double, DOUBLE_TYPE)

#undef INSTANTIATE

value_scalar::value_scalar(numeric_type dtype) : dtype_(dtype) {}
value_scalar::value_scalar(scalar const & s) : dtype_(s.dtype()) { init(s); }
value_scalar::value_scalar(array_expression const &expr) : dtype_(expr.dtype()) { init(scalar(expr)); }

values_holder value_scalar::values() const
{ return values_; }

numeric_type value_scalar::dtype() const
{ return dtype_; }

template<class T>
T value_scalar::cast() const
{
  switch(dtype_)
  {
    case CHAR_TYPE: return values_.int8;
    case UCHAR_TYPE: return values_.uint8;
    case SHORT_TYPE: return values_.int16;
    case USHORT_TYPE: return values_.uint16;
    case INT_TYPE: return values_.int32;
    case UINT_TYPE: return values_.uint32;
    case LONG_TYPE: return values_.int64;
    case ULONG_TYPE: return values_.uint64;
//    case HALF_TYPE: return values_.float16;
    case FLOAT_TYPE: return values_.float32;
    case DOUBLE_TYPE: return values_.float64;
    default: throw unknown_datatype(dtype_); //unreachable
  }
}

#define INSTANTIATE(type) value_scalar::operator type() const { return cast<type>(); }
  INSTANTIATE(bool)
  INSTANTIATE(int8_t)
  INSTANTIATE(u_int8_t)
  INSTANTIATE(int16_t)
  INSTANTIATE(u_int16_t)
  INSTANTIATE(int32_t)
  INSTANTIATE(u_int32_t)
  INSTANTIATE(int64_t)
  INSTANTIATE(u_int64_t)
  INSTANTIATE(cl_float)
  INSTANTIATE(cl_double)
#undef INSTANTIATE

value_scalar int8(int8_t v) { return value_scalar(v); }
value_scalar uint8(u_int8_t v) { return value_scalar(v); }
value_scalar int16(int16_t v) { return value_scalar(v); }
value_scalar uint16(u_int16_t v) { return value_scalar(v); }
value_scalar int32(int32_t v) { return value_scalar(v); }
value_scalar uint32(u_int32_t v) { return value_scalar(v); }
value_scalar int64(int64_t v) { return value_scalar(v); }
value_scalar uint64(u_int64_t v) { return value_scalar(v); }
value_scalar float32(cl_float v) { return value_scalar(v); }
value_scalar float64(cl_double v) { return value_scalar(v); }

#define VALUE(type, OP, x, y) (type)x OP y
#define INSTANTIATE(NAME, LDEC, RDEC, OP)\
value_scalar NAME(LDEC, RDEC)\
{\
  switch(x.dtype())\
  {\
  case CHAR_TYPE: return VALUE(int8_t, OP, x, y);\
  case UCHAR_TYPE: return VALUE(u_int8_t, OP, x, y);\
  case SHORT_TYPE: return VALUE(int16_t, OP, x, y);\
  case USHORT_TYPE: return VALUE(u_int16_t, OP, x, y);\
  case INT_TYPE: return VALUE(int32_t, OP, x, y);\
  case UINT_TYPE: return VALUE(u_int32_t, OP, x, y);\
  case LONG_TYPE: return VALUE(int64_t, OP, x, y);\
  case ULONG_TYPE: return VALUE(u_int64_t, OP, x, y);\
  case FLOAT_TYPE: return VALUE(cl_float, OP, x, y);\
  case DOUBLE_TYPE: return VALUE(cl_double, OP, x, y);\
  default: throw unknown_datatype(x.dtype());\
  }\
}

#define INSTANTIATE_ALL(NAME, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, int8_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, u_int8_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, int16_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, u_int16_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, int32_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, u_int32_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, int64_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, u_int64_t y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, cl_float y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, cl_double y, EXPR)\
  INSTANTIATE(NAME, int8_t y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, u_int8_t y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, int16_t y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, u_int16_t y, value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, int32_t y,    value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, u_int32_t y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, int64_t y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, u_int64_t y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, cl_float y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, cl_double y, value_scalar const & x, EXPR)

INSTANTIATE_ALL(operator+, +)
INSTANTIATE_ALL(operator-, -)
INSTANTIATE_ALL(operator*, *)
INSTANTIATE_ALL(operator/, /)
INSTANTIATE_ALL(operator>,  > )
INSTANTIATE_ALL(operator>=, >=)
INSTANTIATE_ALL(operator<,  < )
INSTANTIATE_ALL(operator<=, <=)
INSTANTIATE_ALL(operator==, ==)
INSTANTIATE_ALL(operator!=, !=)

#undef VALUE
#define VALUE(type, OP, x, y) OP((type)x,(type)y)
INSTANTIATE_ALL(max, std::max)
INSTANTIATE_ALL(min, std::min)

#undef VALUE
#define VALUE(type, OP, x, y) OP((type)x, y)
INSTANTIATE_ALL(pow, std::pow)

#undef INSTANTIATE_ALL
#undef INSTANTIATE

std::ostream & operator<<(std::ostream & os, value_scalar const & s)
{
  switch(s.dtype())
  {
    case CHAR_TYPE: return os << static_cast<int8_t>(s);
    case UCHAR_TYPE: return os << static_cast<u_int8_t>(s);
    case SHORT_TYPE: return os << static_cast<int16_t>(s);
    case USHORT_TYPE: return os << static_cast<u_int16_t>(s);
    case INT_TYPE: return os << static_cast<int32_t>(s);
    case UINT_TYPE: return os << static_cast<u_int32_t>(s);
    case LONG_TYPE: return os << static_cast<int64_t>(s);
    case ULONG_TYPE: return os << static_cast<u_int64_t>(s);
    case FLOAT_TYPE: return os << static_cast<cl_float>(s);
    case DOUBLE_TYPE: return os << static_cast<cl_double>(s);
    default: throw unknown_datatype(s.dtype());;
  }
}

}
