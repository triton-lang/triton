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
