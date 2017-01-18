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

#ifndef ISAAC_VALUE_SCALAR_H
#define ISAAC_VALUE_SCALAR_H

#include "isaac/defines.h"
#include "isaac/types.h"
#include "isaac/common/numeric_type.h"
#include <stdint.h>

namespace isaac
{

class scalar;
class expression_tree;

union ISAACAPI values_holder
{
  int8_t int8;
  uint8_t uint8;
  int16_t int16;
  uint16_t uint16;
  int32_t int32;
  uint32_t uint32;
  int64_t int64;
  uint64_t uint64;
  float float32;
  double float64;
};

class ISAACAPI value_scalar
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
  ISAAC_INSTANTIATE(long long)
  ISAAC_INSTANTIATE(unsigned long long)
  ISAAC_INSTANTIATE(float)
  ISAAC_INSTANTIATE(double)
#undef ISAAC_INSTANTIATE
  value_scalar(values_holder values, numeric_type dtype);
  explicit value_scalar(scalar const &);
  explicit value_scalar(expression_tree const &);
  explicit value_scalar(numeric_type dtype = INVALID_NUMERIC_TYPE);

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
  INSTANTIATE(long long)
  INSTANTIATE(unsigned long long)
  INSTANTIATE(float)
  INSTANTIATE(double)
#undef INSTANTIATE
private:
  values_holder values_;
  numeric_type dtype_;
};

ISAACAPI value_scalar int8(int8_t v);
ISAACAPI value_scalar uint8(uint8_t v);
ISAACAPI value_scalar int16(int16_t v);
ISAACAPI value_scalar uint16(uint16_t v);
ISAACAPI value_scalar int32(int32_t v);
ISAACAPI value_scalar uint32(uint32_t v);
ISAACAPI value_scalar int64(int64_t v);
ISAACAPI value_scalar uint64(uint64_t v);
ISAACAPI value_scalar float32(float v);
ISAACAPI value_scalar float64(double v);

template<class T>
ISAACAPI T cast(isaac::value_scalar const &);

#define ISAAC_DECLARE_BINARY_OPERATOR(RET, OPNAME) \
ISAACAPI RET OPNAME (value_scalar const &, char  );\
ISAACAPI RET OPNAME (value_scalar const &, unsigned char );\
ISAACAPI RET OPNAME (value_scalar const &, short );\
ISAACAPI RET OPNAME (value_scalar const &, unsigned short);\
ISAACAPI RET OPNAME (value_scalar const &, int   );\
ISAACAPI RET OPNAME (value_scalar const &, unsigned int  );\
ISAACAPI RET OPNAME (value_scalar const &, long  );\
ISAACAPI RET OPNAME (value_scalar const &, unsigned long );\
ISAACAPI RET OPNAME (value_scalar const &, long long);\
ISAACAPI RET OPNAME (value_scalar const &, unsigned long long);\
ISAACAPI RET OPNAME (value_scalar const &, float );\
ISAACAPI RET OPNAME (value_scalar const &, double);\
ISAACAPI RET OPNAME (char   , value_scalar const &);\
ISAACAPI RET OPNAME (unsigned char  , value_scalar const &);\
ISAACAPI RET OPNAME (short  , value_scalar const &);\
ISAACAPI RET OPNAME (unsigned short , value_scalar const &);\
ISAACAPI RET OPNAME (int    , value_scalar const &);\
ISAACAPI RET OPNAME (unsigned int   , value_scalar const &);\
ISAACAPI RET OPNAME (long   , value_scalar const &);\
ISAACAPI RET OPNAME (unsigned long  , value_scalar const &);\
ISAACAPI RET OPNAME (long long, value_scalar const &);\
ISAACAPI RET OPNAME (unsigned long long, value_scalar const &);\
ISAACAPI RET OPNAME (float  , value_scalar const &);\
ISAACAPI RET OPNAME (double , value_scalar const &);\
ISAACAPI RET OPNAME (value_scalar const &, value_scalar const &);

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

ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, (max))
ISAAC_DECLARE_BINARY_OPERATOR(value_scalar, (min))
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

ISAACAPI std::ostream & operator<<(std::ostream & os, value_scalar const & s);
ISAACAPI value_scalar cast(value_scalar const & in, numeric_type dtype);

}

#endif
