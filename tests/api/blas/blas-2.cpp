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
  T a = 4.2, b = 5.6;
  sc::int_t M = A.shape()[0], N = A.shape()[1];
  simple_vector<T> bufy(M);
  simple_vector<T> bufx(N);
  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(y.context(),0);
  T yi = 0, xi = 0;
  if(y.context().backend()==sc::driver::OPENCL)
  {
      cl_command_queue clqueue = queue.handle().cl();
      ADD_TEST_2D_RD(DT+"GEMV-ROW-N"+ST, M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = a*yi + b*cy[i],
               BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasRowMajor, clblasTrans, N, M, a, cl(A), off(A), ld(A),  cl(x), off(x), inc(x), b, cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL), y, bufy, cy);
      ADD_TEST_2D_RD(DT+"GEMV-ROW-T"+ST, N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = a*xi + b*cx[i],
              BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasRowMajor, clblasNoTrans, N, M, a, cl(A), off(A), ld(A), cl(y), off(y), inc(y), b, cl(x), off(x), inc(x), 1, &clqueue, 0, NULL, NULL), x, bufx, cx);
      ADD_TEST_2D_RD(DT+"GEMV-COL-N"+ST, M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = a*yi + b*cy[i],
              BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasColumnMajor, clblasNoTrans, M, N, a, cl(A), off(A), ld(A), cl(x), off(x), inc(x), b, cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL), y, bufy, cy);
      ADD_TEST_2D_RD(DT+"GEMV-COL-T"+ST, N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = a*xi + b*cx[i],
              BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasColumnMajor, clblasTrans, M, N, a, cl(A), off(A), ld(A), cl(y), off(y), inc(y), b, cl(x), off(x), inc(x), 1, &clqueue, 0, NULL, NULL), x, bufx, cx);
  }
  if(y.context().backend()==sc::driver::CUDA)
  {
      ADD_TEST_2D_RD(DT+"GEMV-N"+ST, M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = a*yi + b*cy[i],
              BLAS<T>::F(cublasSgemv, cublasDgemv)('N', M, N, a, (T*)cu(A) + off(A), ld(A), (T*)cu(x) + off(x), inc(x), b, (T*)cu(y) + off(y), inc(y)), y, bufy, cy);
      ADD_TEST_2D_RD(DT+"GEMV-T"+ST, N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = a*xi + b*cx[i],
              BLAS<T>::F(cublasSgemv, cublasDgemv)('T', M, N, a, (T*)cu(A) + off(A), ld(A), (T*)cu(y) + off(y), inc(y), b, (T*)cu(x) + off(x), inc(x)), x, bufx, cx);
  }
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  typedef std::tuple<int, int> pair;
  auto tp = std::make_tuple<int,int>;
  std::vector<pair> shapes = { tp(173, 241), tp(173, 1), tp(1, 241) };
  int M, N;
  for(pair const & t: shapes){
    std::tie(M, N) = t;
    int SUBM = std::min(7,M), SUBN = std::min(11,N), STRIDEN = std::min(2, N);
    int STARTM = std::min(5,M-1), STARTN = std::min(7, N-1), STRIDEM = std::min(3,M);
    INIT_VECTOR(M, SUBM, STARTM, STRIDEM, cy, y, ctx);
    INIT_VECTOR(N, SUBN, STARTN, STRIDEN, cx, x, ctx);
    INIT_MATRIX(M, SUBM, STARTM, 1, N, SUBN, STARTN, 1, cA, A, ctx);
    std::string suffix = "MN";
    if(M==1) suffix[0] = '1';
    if(N==1) suffix[1] = '1';
    test_impl("FULL-" + suffix, cy, cA, cx, y, A, x, nfail, npass);
    test_impl("SUB-" + suffix, cy_s, cA_s, cx_s, y_s, A_s, x_s, nfail, npass);
  }

}
int main()
{
  clblasSetup();
  int err = run_test(test<float>, test<double>);
  clblasTeardown();
  return err;
}
