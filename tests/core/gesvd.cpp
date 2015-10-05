#include "isaac/symbolic/execute.h"
#include "isaac/symbolic/io.h"

#include "common.hpp"
#include "external/f2c.h"
#include "external/clapack.h"

namespace sc = isaac;

int main()
{
  typedef float T;
  using sc::_i0;

  long int M = 10;
  long int N = 10;
  long int SUBM = 7;
  long int SUBN = 11;

  isaac::driver::Context const & ctx = isaac::driver::backend::contexts::get_default();
  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cA, A, ctx);
  INIT_VECTOR(N-1, SUBN-1, 5, 3, ccos, cos, ctx);
  INIT_VECTOR(M-1, SUBM-1, 5, 3, csin, sin, ctx);

  char side = 'R';
  char pivot = 'V';
  char direct = 'B';
  float* pA = cA_full.data().data();
  float* pcos = ccos_full.data().data();
  float* psin = csin_full.data().data();
  long int lda = cA_full.ld();

  std::cout << A_full << std::endl;
  std::cout << std::endl;

  slasr_(&side, &pivot, &direct, &M, &M, pcos, psin, pA, &lda);
  sc::math_expression tree = sfor(_i0 = 8, _i0 >= 0, _i0-=1, rot(col(A_full, _i0), col(A_full, _i0 + 1), cos_full[_i0], sin_full[_i0]));
  std::cout << to_string(tree) << std::endl;
  sc::execute(tree);

  std::cout << cos_full << std::endl;
  std::cout << sin_full << std::endl;
  std::cout << std::endl;

  std::cout << sc::array(M, N, cA_full.data(), ctx) << std::endl;
  std::cout << std::endl;
  std::cout << A_full << std::endl;
}
