/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_COMMON_NUMERIC_TYPE_H
#define ISAAC_COMMON_NUMERIC_TYPE_H

#include <stdexcept>
#include "isaac/exception/api.h"
#include "simple_half.h"
namespace isaac
{

enum numeric_type
{
  INVALID_NUMERIC_TYPE = 0,
//  BOOL_TYPE,
  CHAR_TYPE,
  UCHAR_TYPE,
  SHORT_TYPE,
  USHORT_TYPE,
  INT_TYPE,
  UINT_TYPE,
  LONG_TYPE,
  ULONG_TYPE,
  HALF_TYPE,
  FLOAT_TYPE,
  DOUBLE_TYPE
};

inline std::string to_string(numeric_type const & type)
{
  switch (type)
  {
//  case BOOL_TYPE: return "bool";
  case CHAR_TYPE: return "char";
  case UCHAR_TYPE: return "uchar";
  case SHORT_TYPE: return "short";
  case USHORT_TYPE: return "ushort";
  case INT_TYPE:  return "int";
  case UINT_TYPE: return "uint";
  case LONG_TYPE:  return "long";
  case ULONG_TYPE: return "ulong";
  case HALF_TYPE : return "half";
  case FLOAT_TYPE : return "float";
  case DOUBLE_TYPE : return "double";
  default : throw unknown_datatype(type);
  }
}

inline numeric_type numeric_type_from_string(std::string const & name)
{
  if(name=="float16") return HALF_TYPE;
  if(name=="float32") return FLOAT_TYPE;
  if(name=="float64") return DOUBLE_TYPE;
  throw std::invalid_argument("Invalid datatype: " + name);
}

inline unsigned int size_of(numeric_type type)
{
  switch (type)
  {
//  case BOOL_TYPE:
  case UCHAR_TYPE:
  case CHAR_TYPE: return 1;

  case HALF_TYPE:
  case USHORT_TYPE:
  case SHORT_TYPE: return 2;

  case UINT_TYPE:
  case INT_TYPE:
  case FLOAT_TYPE: return 4;

  case ULONG_TYPE:
  case LONG_TYPE:
  case DOUBLE_TYPE: return 8;

  default: throw unknown_datatype(type);
  }
}

template<size_t size, bool is_unsigned>
struct to_int_numeric_type_impl;

#define ISAAC_INSTANTIATE_INT_TYPE_IMPL(SIZE, IS_UNSIGNED, TYPE) \
    template<> struct to_int_numeric_type_impl<SIZE, IS_UNSIGNED> { static const numeric_type value = TYPE; }
ISAAC_INSTANTIATE_INT_TYPE_IMPL(1, false, CHAR_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(2, false, SHORT_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(4, false, INT_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(8, false, LONG_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(1, true, UCHAR_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(2, true, USHORT_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(4, true, UINT_TYPE);
ISAAC_INSTANTIATE_INT_TYPE_IMPL(8, true, ULONG_TYPE);
#undef ISAAC_INSTANTIATE_INT_TYPE_IMPL

template<class T>
struct to_int_numeric_type
{
    static const numeric_type value = to_int_numeric_type_impl<sizeof(T), std::is_unsigned<T>::value>::value;
};

template<class T> struct to_numeric_type { static const numeric_type value = to_int_numeric_type<T>::value; };

template<> struct to_numeric_type<char> { static const numeric_type value = CHAR_TYPE; };
template<> struct to_numeric_type<unsigned char> { static const numeric_type value = UCHAR_TYPE ; };
template<> struct to_numeric_type<short> { static const numeric_type value = SHORT_TYPE ; };
template<> struct to_numeric_type<unsigned short> { static const numeric_type value = USHORT_TYPE ; };
template<> struct to_numeric_type<int> { static const numeric_type value = INT_TYPE ; };
template<> struct to_numeric_type<unsigned int> { static const numeric_type value = UINT_TYPE ; };
template<> struct to_numeric_type<long> { static const numeric_type value = LONG_TYPE ; };
template<> struct to_numeric_type<unsigned long> { static const numeric_type value = ULONG_TYPE ; };
template<> struct to_numeric_type<half> { static const numeric_type value = HALF_TYPE; };
template<> struct to_numeric_type<float> { static const numeric_type value = FLOAT_TYPE; };
template<> struct to_numeric_type<double> { static const numeric_type value = DOUBLE_TYPE; };

template<class T> typename std::enable_if<std::is_arithmetic<T>::value, numeric_type>::type numeric_type_of(T) { return to_numeric_type<T>::value; }
template<class T> typename std::enable_if<!std::is_arithmetic<T>::value, numeric_type>::type numeric_type_of(T const & x) { return x.dtype(); }
}

#endif
