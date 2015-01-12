#include "atidlas/value_scalar.h"

namespace atidlas
{

value_scalar::value_scalar(cl_char value) : dtype_(CHAR_TYPE) { values_.int8 = value; }
value_scalar::value_scalar(cl_uchar value) : dtype_(UCHAR_TYPE) { values_.uint8 = value; }
value_scalar::value_scalar(cl_short value) : dtype_(SHORT_TYPE) { values_.int16 = value; }
value_scalar::value_scalar(cl_ushort value) : dtype_(USHORT_TYPE) { values_.uint16 = value; }
value_scalar::value_scalar(cl_int value) : dtype_(INT_TYPE) { values_.int32 = value; }
value_scalar::value_scalar(cl_uint value) : dtype_(UINT_TYPE) { values_.uint32 = value; }
value_scalar::value_scalar(cl_long value) : dtype_(LONG_TYPE) { values_.int64 = value; }
value_scalar::value_scalar(cl_ulong value) : dtype_(ULONG_TYPE) { values_.uint64 = value; }
//value_scalar::value_scalar(cl_half value) : dtype_(HALF_TYPE) { values_.float16 = value; }
value_scalar::value_scalar(cl_float value) : dtype_(FLOAT_TYPE) { values_.float32 = value; }
value_scalar::value_scalar(cl_double value) : dtype_(DOUBLE_TYPE) { values_.float64 = value; }

values_holder value_scalar::values() const
{ return values_; }

numeric_type value_scalar::dtype() const
{ return dtype_; }

template<class T>
T value_scalar::value() const
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
    default: throw; //unreachable
  }
}

template cl_char value_scalar::value<cl_char>() const;
template cl_uchar value_scalar::value<cl_uchar>() const;
template cl_short value_scalar::value<cl_short>() const;
template cl_ushort value_scalar::value<cl_ushort>() const;
template cl_int value_scalar::value<cl_int>() const;
template cl_uint value_scalar::value<cl_uint>() const;
template cl_long value_scalar::value<cl_long>() const;
template cl_ulong value_scalar::value<cl_ulong>() const;
template cl_float value_scalar::value<cl_float>() const;
template cl_double value_scalar::value<cl_double>() const;

value_scalar int8(cl_char v) { return value_scalar(v); }
value_scalar uint8(cl_uchar v) { return value_scalar(v); }
value_scalar int16(cl_short v) { return value_scalar(v); }
value_scalar uint16(cl_ushort v) { return value_scalar(v); }
value_scalar int32(cl_int v) { return value_scalar(v); }
value_scalar uint32(cl_uint v) { return value_scalar(v); }
value_scalar int64(cl_long v) { return value_scalar(v); }
value_scalar uint64(cl_ulong v) { return value_scalar(v); }
value_scalar float32(cl_float v) { return value_scalar(v); }
value_scalar float64(cl_double v) { return value_scalar(v); }

}
