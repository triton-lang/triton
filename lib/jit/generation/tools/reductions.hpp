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
