#ifndef ATIDLAS_TRAITS_SIZE_HPP_
#define ATIDLAS_TRAITS_SIZE_HPP_

#include "atidlas/forwards.h"
#include "atidlas/tools/predicate.hpp"
#include <vector>

namespace atidlas
{
namespace traits
{

template<typename LHS>
atidlas_int_t size(vector_expression<LHS, const int, op_matrix_diag> const & proxy)
{
  int k = proxy.rhs();
  int A_size1 = static_cast<int>(size1(proxy.lhs()));
  int A_size2 = static_cast<int>(size2(proxy.lhs()));

  int row_depth = std::min(A_size1, A_size1 + k);
  int col_depth = std::min(A_size2, A_size2 - k);

  return atidlas_int_t(std::min(row_depth, col_depth));
}

template<typename LHS>
atidlas_int_t size(vector_expression<LHS, const unsigned int, op_row> const & proxy)
{ return size2(proxy.lhs());}

template<typename LHS>
atidlas_int_t size(vector_expression<LHS, const unsigned int, op_column> const & proxy)
{ return size1(proxy.lhs());}

inline atidlas_int_t size(vector_base const & x)
{ return x.size(); }



}
}


#endif
