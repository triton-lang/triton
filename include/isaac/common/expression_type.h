#ifndef ISAAC_COMMON_EXPRESSION_TYPE_H
#define ISAAC_COMMON_EXPRESSION_TYPE_H

#include <string>
#include <stdexcept>

namespace isaac
{

enum expression_type
{
  INVALID_EXPRESSION_TYPE,
  AXPY_TYPE,
  GER_TYPE,
  DOT_TYPE,
  GEMV_N_TYPE,
  GEMV_T_TYPE,
  GEMM_NN_TYPE,
  GEMM_TN_TYPE,
  GEMM_NT_TYPE,
  GEMM_TT_TYPE
};

inline expression_type expression_type_from_string(std::string const & name)
{
  if(name=="axpy") return AXPY_TYPE;
  if(name=="dot") return DOT_TYPE;
  if(name=="ger") return GER_TYPE;
  if(name=="gemv_n") return GEMV_N_TYPE;
  if(name=="gemv_t") return GEMV_T_TYPE;
  if(name=="gemm_nn") return GEMM_NN_TYPE;
  if(name=="gemm_nt") return GEMM_NT_TYPE;
  if(name=="gemm_tn") return GEMM_TN_TYPE;
  if(name=="gemm_tt") return GEMM_TT_TYPE;
  throw std::invalid_argument("Unrecognized expression: " + name);
}


}

#endif
