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
  MATRIX_PRODUCT_NN,
  MATRIX_PRODUCT_TN,
  MATRIX_PRODUCT_NT,
  MATRIX_PRODUCT_TT
};

inline expression_type expression_type_from_string(std::string const & name)
{
  if(name=="elementwise_1d") return ELEMENTWISE_1D;
  if(name=="reduce_1d") return REDUCE_1D;
  if(name=="elementwise_2d") return ELEMENTWISE_2D;
  if(name=="reduce_2d_rows") return REDUCE_2D_ROWS;
  if(name=="reduce_2d_cols") return REDUCE_2D_COLS;
  if(name=="matrix_product_nn") return MATRIX_PRODUCT_NN;
  if(name=="matrix_product_nt") return MATRIX_PRODUCT_NT;
  if(name=="matrix_product_tn") return MATRIX_PRODUCT_TN;
  if(name=="matrix_product_tt") return MATRIX_PRODUCT_TT;
  throw std::invalid_argument("Unrecognized expression: " + name);
}


}

#endif
