#include <cmath>
#include <iostream>
#include <algorithm>
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
    case CHAR_TYPE: values_.int8 = (int8_t)s; break;
    case UCHAR_TYPE: values_.uint8 = (uint8_t)s; break;
    case SHORT_TYPE: values_.int16 = (int16_t)s; break;
    case USHORT_TYPE: values_.uint16 = (uint16_t)s; break;
    case INT_TYPE: values_.int32 = (int32_t)s; break;
    case UINT_TYPE: values_.uint32 = (uint32_t)s; break;
    case LONG_TYPE: values_.int64 = (int64_t)s; break;
    case ULONG_TYPE: values_.uint64 = (uint64_t)s; break;
//    case HALF_TYPE: values_.float16 = s; break;
    case FLOAT_TYPE: values_.float32 = (float)s; break;
    case DOUBLE_TYPE: values_.float64 = (double)s; break;
    default: throw unknown_datatype(dtype_);
  }
}

#define INSTANTIATE(CLTYPE) value_scalar::value_scalar(CLTYPE value, numeric_type dtype) : dtype_(dtype) { init(value); }

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

value_scalar::value_scalar(numeric_type dtype) : dtype_(dtype) {}
value_scalar::value_scalar(scalar const & s) : dtype_(s.dtype()) { s.inject(values_); }
value_scalar::value_scalar(array_expression const &expr) : dtype_(expr.dtype()) { scalar(expr).inject(values_); }
value_scalar::value_scalar(values_holder values, numeric_type dtype) : values_(values), dtype_(dtype) {}
values_holder value_scalar::values() const
{ return values_; }

numeric_type value_scalar::dtype() const
{ return dtype_; }

template<class T>
T value_scalar::cast() const
{
  switch(dtype_)
  {
    case CHAR_TYPE: return static_cast<T>(values_.int8);
    case UCHAR_TYPE: return static_cast<T>(values_.uint8);
    case SHORT_TYPE: return static_cast<T>(values_.int16);
    case USHORT_TYPE: return static_cast<T>(values_.uint16);
    case INT_TYPE: return static_cast<T>(values_.int32);
    case UINT_TYPE: return static_cast<T>(values_.uint32);
    case LONG_TYPE: return static_cast<T>(values_.int64);
    case ULONG_TYPE: return static_cast<T>(values_.uint64);
    case FLOAT_TYPE: return static_cast<T>(values_.float32);
    case DOUBLE_TYPE: return static_cast<T>(values_.float64);
    default: throw unknown_datatype(dtype_); //unreachable
  }
}

#define INSTANTIATE(type) value_scalar::operator type() const { return cast<type>(); }
  //INSTANTIATE(bool)
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

value_scalar int8(int8_t v) { return value_scalar(v); }
value_scalar uint8(uint8_t v) { return value_scalar(v); }
value_scalar int16(int16_t v) { return value_scalar(v); }
value_scalar uint16(uint16_t v) { return value_scalar(v); }
value_scalar int32(int32_t v) { return value_scalar(v); }
value_scalar uint32(uint32_t v) { return value_scalar(v); }
value_scalar int64(int64_t v) { return value_scalar(v); }
value_scalar uint64(uint64_t v) { return value_scalar(v); }
value_scalar float32(float v) { return value_scalar(v); }
value_scalar float64(double v) { return value_scalar(v); }

#define VALUE(type, OP, x, y) (type)x OP y
#define INSTANTIATE(NAME, LDEC, RDEC, OP)\
value_scalar NAME(LDEC, RDEC)\
{\
  switch(x.dtype())\
  {\
  case CHAR_TYPE: return VALUE(char, OP, x, y);\
  case UCHAR_TYPE: return VALUE(unsigned char, OP, x, y);\
  case SHORT_TYPE: return VALUE(short, OP, x, y);\
  case USHORT_TYPE: return VALUE(unsigned short, OP, x, y);\
  case INT_TYPE: return VALUE(int, OP, x, y);\
  case UINT_TYPE: return VALUE(unsigned int, OP, x, y);\
  case LONG_TYPE: return VALUE(long, OP, x, y);\
  case ULONG_TYPE: return VALUE(unsigned long, OP, x, y);\
  case FLOAT_TYPE: return VALUE(float, OP, x, y);\
  case DOUBLE_TYPE: return VALUE(double, OP, x, y);\
  default: throw unknown_datatype(x.dtype());\
  }\
}

#define INSTANTIATE_ALL(NAME, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, char y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, unsigned char y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, short y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, unsigned short y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, int y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, unsigned int y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, long y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, unsigned long y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, long long y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, unsigned long long y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, float y, EXPR)\
  INSTANTIATE(NAME, value_scalar const & x, double y, EXPR)\
  INSTANTIATE(NAME, char y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, unsigned char y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, short y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, unsigned short y, value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, int y,    value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, unsigned int y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, long y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, unsigned long y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, long long y,   value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, unsigned long long y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, float y,  value_scalar const & x, EXPR)\
  INSTANTIATE(NAME, double y, value_scalar const & x, EXPR)

INSTANTIATE_ALL(operator+, +)
INSTANTIATE_ALL(operator-, -)
INSTANTIATE_ALL(operator*, *)
INSTANTIATE_ALL(operator/, /)

//INSTANTIATE_ALL(operator>,  > )
//INSTANTIATE_ALL(operator>=, >=)
//INSTANTIATE_ALL(operator<,  < )
//INSTANTIATE_ALL(operator<=, <=)
//INSTANTIATE_ALL(operator==, ==)
//INSTANTIATE_ALL(operator!=, !=)

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
    case CHAR_TYPE: return os << static_cast<char>(s);
    case UCHAR_TYPE: return os << static_cast<unsigned char>(s);
    case SHORT_TYPE: return os << static_cast<short>(s);
    case USHORT_TYPE: return os << static_cast<unsigned short>(s);
    case INT_TYPE: return os << static_cast<int>(s);
    case UINT_TYPE: return os << static_cast<unsigned int>(s);
    case LONG_TYPE: return os << static_cast<long>(s);
    case ULONG_TYPE: return os << static_cast<unsigned long>(s);
    case FLOAT_TYPE: return os << static_cast<float>(s);
    case DOUBLE_TYPE: return os << static_cast<double>(s);
    default: throw unknown_datatype(s.dtype());;
  }
}

}
