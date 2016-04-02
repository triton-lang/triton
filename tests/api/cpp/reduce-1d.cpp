#include <cmath>
#include <iostream>

#include "api.hpp"
#include "isaac/array.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;
typedef sc::int_t int_t;

template<typename T>
void test_impl(std::string const & ST, simple_vector_base<T> & cx, simple_vector_base<T> & cy,
                                sc::array_base & x, sc::array_base & y, int& nfail, int& npass)
{
  std::string DT = std::is_same<T, float>::value?"S":"D";
  std::string PFX = "[" + DT + "," + ST + "]";
  sc::driver::Context const & context = x.context();
  T cs = 0;
  sc::scalar ds(cs, context);
  int_t N = cx.size();
  sc::array scratch(N, x.dtype());

  ADD_TEST_1D_RD(PFX + " s = x'.y", cs+=cx[i]*cy[i], 0, cs, ds = dot(x,y));
  ADD_TEST_1D_RD(PFX + " s = exp(x'.y)", cs += cx[i]*cy[i], 0, std::exp(cs), ds = exp(dot(x,y)));
  ADD_TEST_1D_RD(PFX + " s = 1 + x'.y", cs += cx[i]*cy[i], 0, 1 + cs, ds = 1 + dot(x,y));
  ADD_TEST_1D_RD(PFX + " s = x'.y + y'.y", cs+= cx[i]*cy[i] + cy[i]*cy[i], 0, cs, ds = dot(x,y) + dot(y,y));
  ADD_TEST_1D_RD(PFX + " s = max(x)", cs = std::max(cs, cx[i]), std::numeric_limits<T>::min(), cs, ds = max(x));
  ADD_TEST_1D_RD(PFX + " s = min(x)", cs = std::min(cs, cx[i]), std::numeric_limits<T>::max(), cs, ds = min(x));
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  int_t N = 10007;
  int_t SUBN = 7;
  INIT_VECTOR(N, SUBN, 0, 1, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 0, 1, cy, y, ctx);
  test_impl("FULL", cx, cy, x, y, nfail, npass);
  test_impl("SUB", cx_s, cy_s, x_s, y_s, nfail, npass);
}


int main()
{
  return run_test(test<float>, test<double>);
}
