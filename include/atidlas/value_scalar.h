#ifndef ATIDLAS_VALUE_SCALAR_H
#define ATIDLAS_VALUE_SCALAR_H

#include "atidlas/types.h"
#include "atidlas/cl/cl.hpp"

namespace atidlas
{

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

  values_holder values() const;
  template<class T> T value() const;

  numeric_type dtype() const;
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

}

#endif
