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
void test_impl(std::string const & ST, simple_vector_base<T> & cx, simple_vector_base<T>& cy, sc::array_base& x, sc::array_base& y, int& nfail, int& npass)
{
  std::string DT = std::is_same<T, float>::value?"S":"D";
  T a = -4.3;
  int_t N = cy.size();
  sc::driver::CommandQueue & queue = sc::driver::backend::queues::get(x.context(),0);
  T cs = 0;
  sc::scalar ds(cs, y.context());
  sc::array scratch(N, y.dtype());

  if(queue.device().backend()==sc::driver::OPENCL)
  {
      cl_command_queue clqueue = queue.handle().cl();
      ADD_TEST_1D_EW(DT+"AXPY"+ST, cy[i] = a*cx[i] + cy[i], BLAS<T>::F(clblasSaxpy, clblasDaxpy)(N, a, cl(x), off(x), inc(x), cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_1D_EW(DT+"COPY"+ST, cy[i] = cx[i], BLAS<T>::F(clblasScopy, clblasDcopy)(N, cl(x), off(x), inc(x),  cl(y), off(y), inc(y),  1, &clqueue, 0, NULL, NULL));
      ADD_TEST_1D_EW(DT+"SCAL"+ST, cy[i] = a*cy[i], BLAS<T>::F(clblasSscal, clblasDscal)(N, a, cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_1D_RD(DT+"DOT"+ST, cs+=cx[i]*cy[i], 0, cs, BLAS<T>::F(clblasSdot, clblasDdot)(N, cl(ds), 0, cl(x), off(x), inc(x), cl(y), off(y), inc(y),  cl(scratch), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_1D_RD(DT+"ASUM"+ST, cs+=std::fabs(cx[i]), 0, cs, BLAS<T>::F(clblasSasum, clblasDasum)(N, cl(ds), 0, cl(x), off(x), inc(x), cl(scratch), 1, &clqueue, 0, NULL, NULL));
  }
  if(queue.device().backend()==sc::driver::CUDA)
  {
      ADD_TEST_1D_EW(DT+"AXPY"+ST, cy[i] = a*cx[i] + cy[i], BLAS<T>::F(cublasSaxpy, cublasDaxpy)(N, a, (T*)cu(x) + off(x), inc(x), (T*)cu(y) + off(y), inc(y)));
      ADD_TEST_1D_EW(DT+"COPY"+ST, cy[i] = cx[i], BLAS<T>::F(cublasScopy, cublasDcopy)(N, (T*)cu(x) + off(x), inc(x), (T*)cu(y) + off(y), inc(y)));
      ADD_TEST_1D_EW(DT+"SCAL"+ST, cy[i] = a*cy[i], BLAS<T>::F(cublasSscal, cublasDscal)(N, a, (T*)cu(y) + off(y), inc(y)));
      ADD_TEST_1D_RD(DT+"DOT"+ST, cs+=cx[i]*cy[i], 0, cs, ds = BLAS<T>::F(cublasSdot, cublasDdot)(N, (T*)cu(x) + off(x), inc(x), (T*)cu(y) + off(y), inc(y)));
      ADD_TEST_1D_RD(DT+"ASUM"+ST, cs+=std::fabs(cx[i]), 0, cs, ds = BLAS<T>::F(cublasSasum, cublasDasum)(N, (T*)cu(x) + off(x), inc(x)));
  }

}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  int_t N = 10007;
  int_t SUBN = 7;
  INIT_VECTOR(N, SUBN, 3, 11, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 3, 11, cy, y, ctx);
  test_impl("FULL", cx, cy, x, y, nfail, npass);
  test_impl("SUB", cx_s, cy_s, x_s, y_s, nfail, npass);
}

int main()
{
  clblasSetup();
  int err = run_test(test<float>, test<double>);
  clblasTeardown();
  return err;
}
