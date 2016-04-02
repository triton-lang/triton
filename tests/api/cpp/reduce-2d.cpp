#include <cmath>
#include <iostream>
#include <type_traits>
#include "api.hpp"
#include "isaac/array.h"
#include "isaac/driver/common.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;
typedef isaac::int_t int_t;

template<typename T>
void test_impl(std::string const & ST, simple_vector_base<T> & cy, simple_matrix_base<T> const & cA, simple_vector_base<T> & cx,
                                        sc::array_base & y, sc::array_base const & A, sc::array_base & x, int& nfail, int& npass)
{
  std::string DT = std::is_same<T, float>::value?"S":"D";
  std::string PFX = "[" + DT + "," + ST + "]";
  sc::int_t M = A.shape()[0], N = A.shape()[1];
  T yi = 0, xi = 0;
  simple_vector<T> bufy(M);
  simple_vector<T> bufx(N);

   ADD_TEST_2D_RD(PFX + " x = dot(A.T, y)", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = xi, x = dot(trans(A),y), x, bufx, cx);
   ADD_TEST_2D_RD(PFX + " x = sum(A, 0)", N, M, 0, xi+=cA(j,i), cx[i] = xi, x = sum(A,0), x, bufx, cx);
   ADD_TEST_2D_RD(PFX + " x = max(A, 0)", N, M, std::numeric_limits<T>::min(), xi=std::max(xi,cA(j,i)), cx[i] = xi, x = max(A,0), x, bufx, cx);
   ADD_TEST_2D_RD(PFX + " x = min(A, 0)", N, M, std::numeric_limits<T>::max(), xi=std::min(xi,cA(j,i)), cx[i] = xi, x = min(A,0), x, bufx, cx);

   ADD_TEST_2D_RD(PFX + " y = dot(A, x)", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = yi, y = dot(A,x), y, bufy, cy);
   ADD_TEST_2D_RD(PFX + " y = sum(A, 1)", M, N, 0, yi+=cA(i,j), cy[i] = yi, y = sum(A,1), y, bufy, cy);
   ADD_TEST_2D_RD(PFX + " y = max(A, 1)", M, N, std::numeric_limits<T>::min(), yi=std::max(yi,cA(i,j)), cy[i] = yi, y = max(A,1), y, bufy, cy);
   ADD_TEST_2D_RD(PFX + " y = min(A, 1)", M, N, std::numeric_limits<T>::max(), yi=std::min(yi,cA(i,j)), cy[i] = yi, y = min(A,1), y, bufy, cy);
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  int_t M = 173, N = 241;
  int_t SUBM = 7, SUBN = 11;

  INIT_VECTOR(M, SUBM, 7, 2, cy, y, ctx);
  INIT_VECTOR(N, SUBN, 5, 3, cx, x, ctx);
  INIT_MATRIX(M, SUBM, 9, 5, N, SUBN, 8, 4, cA, A, ctx);
  test_impl("FULL", cy, cA, cx, y, A, x, nfail, npass);
  test_impl("SUB", cy_s, cA_s, cx_s, y_s, A_s, x_s, nfail, npass);
}
int main()
{
  return run_test(test<float>, test<double>);
}

