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

#ifndef ISAAC_COMMON_EXPRESSION_TYPE_H
#define ISAAC_COMMON_EXPRESSION_TYPE_H

#include <string>
#include <stdexcept>

namespace isaac
{

enum expression_type
{
  INVALID_EXPRESSION_TYPE,
  ELEMENTWISE_1D,
  ELEMENTWISE_2D,
  REDUCE_1D,
  REDUCE_2D_ROWS,
  REDUCE_2D_COLS,
  GEMM_NN,
  GEMM_TN,
  GEMM_NT,
  GEMM_TT
};

inline expression_type expression_type_from_string(std::string const & name)
{
  if(name=="elementwise_1d") return ELEMENTWISE_1D;
  if(name=="reduce_1d") return REDUCE_1D;
  if(name=="elementwise_2d") return ELEMENTWISE_2D;
  if(name=="reduce_2d_rows") return REDUCE_2D_ROWS;
  if(name=="reduce_2d_cols") return REDUCE_2D_COLS;
  if(name=="gemm_nn") return GEMM_NN;
  if(name=="gemm_nt") return GEMM_NT;
  if(name=="gemm_tn") return GEMM_TN;
  if(name=="gemm_tt") return GEMM_TT;
  throw std::invalid_argument("Unrecognized expression: " + name);
}


}

#endif
