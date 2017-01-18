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

#ifndef _ISAAC_SYMBOLIC_OPERATIONS_H
#define _ISAAC_SYMBOLIC_OPERATIONS_H

#include <string>

namespace isaac
{


/** @brief Optimization enum for grouping operations into unary or binary operations. Just for optimization of lookups. */
enum operation_type_family
{
  INVALID_ = 0,

  // BLAS1-type
  UNARY_ARITHMETIC,
  BINARY_ARITHMETIC,
  REDUCE,

  // BLAS2-type
  REDUCE_ROWS,
  REDUCE_COLUMNS,

  // BLAS3-type
  GEMM
};

/** @brief Enumeration for identifying the possible operations */
enum operation_type
{
  INVALID_TYPE = 0,

  // unary operator
  MINUS_TYPE,
  NEGATE_TYPE,

  // unary expression
  CAST_BOOL_TYPE,
  CAST_CHAR_TYPE,
  CAST_UCHAR_TYPE,
  CAST_SHORT_TYPE,
  CAST_USHORT_TYPE,
  CAST_INT_TYPE,
  CAST_UINT_TYPE,
  CAST_LONG_TYPE,
  CAST_ULONG_TYPE,
  CAST_HALF_TYPE,
  CAST_FLOAT_TYPE,
  CAST_DOUBLE_TYPE,

  ABS_TYPE,
  ACOS_TYPE,
  ASIN_TYPE,
  ATAN_TYPE,
  CEIL_TYPE,
  COS_TYPE,
  COSH_TYPE,
  EXP_TYPE,
  FABS_TYPE,
  FLOOR_TYPE,
  LOG_TYPE,
  LOG10_TYPE,
  SIN_TYPE,
  SINH_TYPE,
  SQRT_TYPE,
  TAN_TYPE,
  TANH_TYPE,
  TRANS_TYPE,

  // binary expression
  ASSIGN_TYPE,
  INPLACE_ADD_TYPE,
  INPLACE_SUB_TYPE,
  ADD_TYPE,
  SUB_TYPE,
  MULT_TYPE,
  DIV_TYPE,
  ELEMENT_ARGFMAX_TYPE,
  ELEMENT_ARGFMIN_TYPE,
  ELEMENT_ARGMAX_TYPE,
  ELEMENT_ARGMIN_TYPE,
  ELEMENT_PROD_TYPE,
  ELEMENT_DIV_TYPE,
  ELEMENT_EQ_TYPE,
  ELEMENT_NEQ_TYPE,
  ELEMENT_GREATER_TYPE,
  ELEMENT_GEQ_TYPE,
  ELEMENT_LESS_TYPE,
  ELEMENT_LEQ_TYPE,
  ELEMENT_POW_TYPE,
  ELEMENT_FMAX_TYPE,
  ELEMENT_FMIN_TYPE,
  ELEMENT_MAX_TYPE,
  ELEMENT_MIN_TYPE,

  //Products
  OUTER_PROD_TYPE,
  GEMM_NN_TYPE,
  GEMM_TN_TYPE,
  GEMM_NT_TYPE,
  GEMM_TT_TYPE,

  //Access modifiers
  RESHAPE_TYPE,
  SHIFT_TYPE,
  DIAG_MATRIX_TYPE,
  DIAG_VECTOR_TYPE,
  ACCESS_INDEX_TYPE,


  PAIR_TYPE,

  OPERATOR_FUSE,
  SFOR_TYPE,
};

struct op_element
{
  op_element();
  op_element(operation_type_family const & _type_family, operation_type const & _type);
  operation_type_family   type_family;
  operation_type          type;
};

std::string to_string(operation_type type);

bool is_assignment(operation_type op);
bool is_operator(operation_type op);
bool is_function(operation_type op);
bool is_cast(operation_type op);
bool is_indexing(operation_type op);

}

#endif
