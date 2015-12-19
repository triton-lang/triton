#include <string>
#include <stdexcept>

#include "isaac/driver/common.h"
#include "isaac/kernels/keywords.h"
#include "isaac/kernels/stream.h"
#include "isaac/symbolic/expression.h"
#include "isaac/types.h"

namespace isaac
{
namespace templates
{

inline void compute_reduce_1d(kernel_generation_stream & os, std::string acc, std::string cur, op_element const & op)
{
  if (detail::is_elementwise_function(op))
    os << acc << "=" << evaluate(op.type) << "(" << acc << "," << cur << ");" << std::endl;
  else
    os << acc << "= (" << acc << ")" << evaluate(op.type)  << "(" << cur << ");" << std::endl;
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

inline bool is_reduce_1d(expression_tree::node const & node)
{
  return node.op.type_family==VECTOR_DOT_TYPE_FAMILY
      || node.op.type_family==COLUMNS_DOT_TYPE_FAMILY
      || node.op.type_family==ROWS_DOT_TYPE_FAMILY;
}


inline bool is_index_reduction(op_element const & op)
{
  return op.type==ELEMENT_ARGFMAX_TYPE
      || op.type==ELEMENT_ARGMAX_TYPE
      || op.type==ELEMENT_ARGFMIN_TYPE
      || op.type==ELEMENT_ARGMIN_TYPE;
}


}

}
