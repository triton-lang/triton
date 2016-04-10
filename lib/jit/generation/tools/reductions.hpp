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

#include <string>
#include <stdexcept>

#include "isaac/driver/common.h"
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/jit/generation/engine/stream.h"
#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/jit/syntax/engine/object.h"
#include "isaac/types.h"

namespace isaac
{
namespace templates
{

inline void compute_reduce_1d(kernel_generation_stream & os, std::string acc, std::string cur, op_element const & op)
{
  if (is_function(op.type))
    os << acc << "=" << to_string(op.type) << "(" << acc << "," << cur << ");" << std::endl;
  else
    os << acc << "= (" << acc << ")" << to_string(op.type)  << "(" << cur << ");" << std::endl;
}

inline void compute_index_reduce_1d(kernel_generation_stream & os, std::string acc, std::string cur, std::string const & acc_value, std::string const & cur_value, op_element const & op)
{
  //        os << acc << " = " << cur_value << ">" << acc_value  << "?" << cur << ":" << acc << ";" << std::endl;
  os << acc << "= select(" << acc << "," << cur << "," << cur_value << ">" << acc_value << ");" << std::endl;
  os << acc_value << "=";
  if (op.type==ELEMENT_ARGFMAX_TYPE) os << "fmax";
  if (op.type==ELEMENT_ARGMAX_TYPE) os << "max";
  if (op.type==ELEMENT_ARGFMIN_TYPE) os << "fmin";
  if (op.type==ELEMENT_ARGMIN_TYPE) os << "min";
  os << "(" << acc_value << "," << cur_value << ");"<< std::endl;
}

inline std::string neutral_element(op_element const & op, driver::backend_type backend, std::string const & dtype)
{
  std::string INF = Infinity(backend, dtype).get();
  std::string N_INF = "-" + INF;

  switch (op.type)
  {
  case ADD_TYPE : return "0";
  case MULT_TYPE : return "1";
  case DIV_TYPE : return "1";
  case ELEMENT_FMAX_TYPE : return N_INF;
  case ELEMENT_ARGFMAX_TYPE : return N_INF;
  case ELEMENT_MAX_TYPE : return N_INF;
  case ELEMENT_ARGMAX_TYPE : return N_INF;
  case ELEMENT_FMIN_TYPE : return INF;
  case ELEMENT_ARGFMIN_TYPE : return INF;
  case ELEMENT_MIN_TYPE : return INF;
  case ELEMENT_ARGMIN_TYPE : return INF;

  default: throw std::runtime_error("Unsupported reduce_1d operator : no neutral element known");
  }
}

}

}
