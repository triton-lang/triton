#ifndef ATIDLAS_TYPES_H
#define ATIDLAS_TYPES_H

#include <CL/cl.hpp>
#include "atidlas/exception/unknown_datatype.h"

namespace atidlas
{

typedef int int_t;

struct size4
{
  size4(int_t s1, int_t s2 = 1) : _1(s1), _2(s2){ }
  int_t prod() const { return _1*_2; }
  bool operator==(size4 const & other) const { return _1==other._1 && _2==other._2; }
  int_t _1;
  int_t _2;
};
inline int_t prod(size4 const & s) { return s._1*s._2; }
inline int_t max(size4 const & s) { return std::max(s._1, s._2); }
inline int_t min(size4 const & s) { return std::min(s._1, s._2); }

struct repeat_infos
{
  int_t sub1;
  int_t sub2;
  int_t rep1;
  int_t rep2;
};


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
//  HALF_TYPE,
  FLOAT_TYPE,
  DOUBLE_TYPE
};

struct array_infos
{
  numeric_type dtype;
  cl_mem data;
  int_t shape1;
  int_t shape2;
  int_t start1;
  int_t start2;
  int_t stride1;
  int_t stride2;
  int_t ld;
};

inline std::string numeric_type_to_string(numeric_type const & type)
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
//  case HALF_TYPE : return "half";
  case FLOAT_TYPE : return "float";
  case DOUBLE_TYPE : return "double";
  default : throw unknown_datatype(type);
  }
}

template<class T> struct to_numeric_type;
//template<> struct to_numeric_type<cl_bool> { static const numeric_type value = BOOL_TYPE; };
template<> struct to_numeric_type<cl_char> { static const numeric_type value = CHAR_TYPE; };
template<> struct to_numeric_type<cl_uchar> { static const numeric_type value = UCHAR_TYPE; };
template<> struct to_numeric_type<cl_short> { static const numeric_type value = SHORT_TYPE; };
template<> struct to_numeric_type<cl_ushort> { static const numeric_type value = USHORT_TYPE; };
template<> struct to_numeric_type<cl_int> { static const numeric_type value = INT_TYPE; };
template<> struct to_numeric_type<cl_uint> { static const numeric_type value = UINT_TYPE; };
template<> struct to_numeric_type<cl_long> { static const numeric_type value = LONG_TYPE; };
template<> struct to_numeric_type<cl_ulong> { static const numeric_type value = ULONG_TYPE; };
//template<> struct to_numeric_type<cl_float> { static const numeric_type value = HALF_TYPE; };
template<> struct to_numeric_type<cl_float> { static const numeric_type value = FLOAT_TYPE; };
template<> struct to_numeric_type<cl_double> { static const numeric_type value = DOUBLE_TYPE; };

inline unsigned int size_of(numeric_type type)
{
  switch (type)
  {
//  case BOOL_TYPE:
  case UCHAR_TYPE:
  case CHAR_TYPE: return 1;

//  case HALF_TYPE:
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

enum expression_type
{
  SCALAR_AXPY_TYPE,
  VECTOR_AXPY_TYPE,
  MATRIX_AXPY_TYPE,
  REDUCTION_TYPE,
  ROW_WISE_REDUCTION_TYPE,
  COL_WISE_REDUCTION_TYPE,
  MATRIX_PRODUCT_NN_TYPE,
  MATRIX_PRODUCT_TN_TYPE,
  MATRIX_PRODUCT_NT_TYPE,
  MATRIX_PRODUCT_TT_TYPE,
  INVALID_EXPRESSION_TYPE
};

struct slice
{
  slice(int_t _start, int_t _end, int_t _stride = 1) : start(_start), size((_end - _start)/_stride), stride(_stride) { }
  int_t start;
  int_t size;
  int_t stride;
};
typedef slice _;

class obj_base{};

}
#endif
