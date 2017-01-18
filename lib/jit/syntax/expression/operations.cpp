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

#include <cstring>

#include "isaac/tools/cpp/string.hpp"

#include "isaac/array.h"
#include "isaac/exception/api.h"

namespace isaac
{

op_element::op_element() {}

op_element::op_element(operation_type_family const & _type_family, operation_type const & _type) : type_family(_type_family), type(_type){}

std::string to_string(operation_type type)
{
  switch (type)
  {
    //Function
    case ABS_TYPE : return "abs";
    case ACOS_TYPE : return "acos";
    case ASIN_TYPE : return "asin";
    case ATAN_TYPE : return "atan";
    case CEIL_TYPE : return "ceil";
    case COS_TYPE : return "cos";
    case COSH_TYPE : return "cosh";
    case EXP_TYPE : return "exp";
    case FABS_TYPE : return "fabs";
    case FLOOR_TYPE : return "floor";
    case LOG_TYPE : return "log";
    case LOG10_TYPE : return "log10";
    case SIN_TYPE : return "sin";
    case SINH_TYPE : return "sinh";
    case SQRT_TYPE : return "sqrt";
    case TAN_TYPE : return "tan";
    case TANH_TYPE : return "tanh";

    case ELEMENT_ARGFMAX_TYPE : return "argfmax";
    case ELEMENT_ARGMAX_TYPE : return "argmax";
    case ELEMENT_ARGFMIN_TYPE : return "argfmin";
    case ELEMENT_ARGMIN_TYPE : return "argmin";
    case ELEMENT_POW_TYPE : return "pow";

    //Arithmetic
    case MINUS_TYPE : return "-";
    case ASSIGN_TYPE : return "=";
    case INPLACE_ADD_TYPE : return "+=";
    case INPLACE_SUB_TYPE : return "-=";
    case ADD_TYPE : return "+";
    case SUB_TYPE : return "-";
    case MULT_TYPE : return "*";
    case ELEMENT_PROD_TYPE : return "*";
    case DIV_TYPE : return "/";
    case ELEMENT_DIV_TYPE : return "/";

    //Relational
    case NEGATE_TYPE: return "!";
    case ELEMENT_EQ_TYPE : return "==";
    case ELEMENT_NEQ_TYPE : return "!=";
    case ELEMENT_GREATER_TYPE : return ">";
    case ELEMENT_GEQ_TYPE : return ">=";
    case ELEMENT_LESS_TYPE : return "<";
    case ELEMENT_LEQ_TYPE : return "<=";

    case ELEMENT_FMAX_TYPE : return "fmax";
    case ELEMENT_FMIN_TYPE : return "fmin";
    case ELEMENT_MAX_TYPE : return "max";
    case ELEMENT_MIN_TYPE : return "min";

    case CAST_BOOL_TYPE : return "(bool)";
    case CAST_CHAR_TYPE : return "(char)";
    case CAST_UCHAR_TYPE : return "(uchar)";
    case CAST_SHORT_TYPE : return "(short)";
    case CAST_USHORT_TYPE : return "(ushort)";
    case CAST_INT_TYPE: return "(int)";
    case CAST_UINT_TYPE : return "(uint)";
    case CAST_LONG_TYPE : return "(long)";
    case CAST_ULONG_TYPE : return "(ulong)";
    case CAST_FLOAT_TYPE : return "(float)";
    case CAST_DOUBLE_TYPE : return "(double)";

    //dot
    case GEMM_NN_TYPE: return "matmatNN";
    case GEMM_NT_TYPE: return "matmatNT";
    case GEMM_TN_TYPE: return "matmatTN";
    case GEMM_TT_TYPE: return "matmatTT";

    //others
    case RESHAPE_TYPE: return "reshape";
    case TRANS_TYPE: return "trans";

    default : throw operation_not_supported_exception("Unsupported operator");
  }
}

bool is_assignment(operation_type op)
{
  return op== ASSIGN_TYPE
      || op== INPLACE_ADD_TYPE
      || op== INPLACE_SUB_TYPE;
}

bool is_operator(operation_type op)
{
  return is_assignment(op)
      || op == ADD_TYPE
      || op == SUB_TYPE
      || op == ELEMENT_PROD_TYPE
      || op == ELEMENT_DIV_TYPE
      || op == MULT_TYPE
      || op == DIV_TYPE
      || op == ELEMENT_EQ_TYPE
      || op == ELEMENT_NEQ_TYPE
      || op == ELEMENT_GREATER_TYPE
      || op == ELEMENT_LESS_TYPE
      || op == ELEMENT_GEQ_TYPE
      || op == ELEMENT_LEQ_TYPE ;
}

bool is_cast(operation_type op)
{
  return op == CAST_BOOL_TYPE
      || op == CAST_CHAR_TYPE
      || op == CAST_UCHAR_TYPE
      || op == CAST_SHORT_TYPE
      || op == CAST_USHORT_TYPE
      || op == CAST_INT_TYPE
      || op == CAST_UINT_TYPE
      || op == CAST_LONG_TYPE
      || op == CAST_ULONG_TYPE
      || op == CAST_FLOAT_TYPE
      || op == CAST_DOUBLE_TYPE
      ;
}

bool is_function(operation_type op)
{
  return is_cast(op)
      || op == ABS_TYPE
      || op == ACOS_TYPE
      || op == ASIN_TYPE
      || op == ATAN_TYPE
      || op == CEIL_TYPE
      || op == COS_TYPE
      || op == COSH_TYPE
      || op == EXP_TYPE
      || op == FABS_TYPE
      || op == FLOOR_TYPE
      || op == LOG_TYPE
      || op == LOG10_TYPE
      || op == SIN_TYPE
      || op == SINH_TYPE
      || op == SQRT_TYPE
      || op == TAN_TYPE
      || op == TANH_TYPE

      || op == ELEMENT_POW_TYPE
      || op == ELEMENT_FMAX_TYPE
      || op == ELEMENT_FMIN_TYPE
      || op == ELEMENT_MAX_TYPE
      || op == ELEMENT_MIN_TYPE;

}

bool is_indexing(operation_type op)
{
  return op == ELEMENT_ARGFMAX_TYPE
      || op == ELEMENT_ARGMAX_TYPE
      || op == ELEMENT_ARGFMIN_TYPE
      || op == ELEMENT_ARGMIN_TYPE;
}

}
