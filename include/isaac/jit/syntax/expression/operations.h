/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
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
