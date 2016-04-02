#include <cmath>
#include "api.hpp"
#include "isaac/array.h"

namespace sc = isaac;
typedef isaac::int_t int_t;

template<typename T>
void test_impl(std::string const & ST, simple_matrix_base<T> & cA, simple_matrix_base<T>& cB, simple_matrix_base<T>& cC, simple_vector_base<T>& cx, simple_vector_base<T>& cy,
          sc::array_base& A, sc::array_base& B, sc::array_base& C, sc::array_base& x, sc::array_base& y, int& nfail, int& npass)
{
  std::string DT = std::is_same<T, float>::value?"S":"D";
  std::string PFX = "[" + DT + "," + ST + "]";
  sc::numeric_type dtype = C.dtype();
  sc::driver::Context const & ctx = C.context();
  int_t M = cC.size1(), N = cC.size2();
  T a = 3.12, b = 3.5;
  sc::scalar da(a, ctx), db(b, ctx);
  simple_vector<T> buffer(M*N);

  ADD_TEST_2D_EW(PFX + " C = A", cC(i,j) = cA(i,j), C = A)
  ADD_TEST_2D_EW(PFX + " C = A + B", cC(i,j) = cA(i,j) + cB(i,j), C = A + B)
  ADD_TEST_2D_EW(PFX + " C = A - B", cC(i,j) = cA(i,j) - cB(i,j), C = A - B)
  ADD_TEST_2D_EW(PFX + " C = A + B + C", cC(i,j) = cA(i,j) + cB(i,j) + cC(i,j), C = A + B + C)

  ADD_TEST_2D_EW(PFX + " C = a*A", cC(i,j) = a*cA(i,j), C = a*A)
  ADD_TEST_2D_EW(PFX + " C = da*A", cC(i,j) = a*cA(i,j), C = da*A)
  ADD_TEST_2D_EW(PFX + " C = a*A + b*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), C= a*A + b*B)
  ADD_TEST_2D_EW(PFX + " C = da*A + b*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), C= da*A + b*B)
  ADD_TEST_2D_EW(PFX + " C = a*A + db*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), C= a*A + db*B)
  ADD_TEST_2D_EW(PFX + " C = da*A + db*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), C= da*A + db*B)

  ADD_TEST_2D_EW(PFX + " C = exp(A)", cC(i,j) = exp(cA(i,j)), C= exp(A))
  ADD_TEST_2D_EW(PFX + " C = abs(A)", cC(i,j) = std::abs(cA(i,j)), C= abs(A))
  ADD_TEST_2D_EW(PFX + " C = acos(A)", cC(i,j) = acos(cA(i,j)), C= acos(A))
  ADD_TEST_2D_EW(PFX + " C = asin(A)", cC(i,j) = asin(cA(i,j)), C= asin(A))
  ADD_TEST_2D_EW(PFX + " C = atan(A)", cC(i,j) = atan(cA(i,j)), C= atan(A))
  ADD_TEST_2D_EW(PFX + " C = ceil(A)", cC(i,j) = ceil(cA(i,j)), C= ceil(A))
  ADD_TEST_2D_EW(PFX + " C = cos(A)", cC(i,j) = cos(cA(i,j)), C= cos(A))
  ADD_TEST_2D_EW(PFX + " C = cosh(A)", cC(i,j) = cosh(cA(i,j)), C= cosh(A))
  ADD_TEST_2D_EW(PFX + " C = floor(A)", cC(i,j) = floor(cA(i,j)), C= floor(A))
  ADD_TEST_2D_EW(PFX + " C = log(A)", cC(i,j) = log(cA(i,j)), C= log(A))
  ADD_TEST_2D_EW(PFX + " C = log10(A)", cC(i,j) = log10(cA(i,j)), C= log10(A))
  ADD_TEST_2D_EW(PFX + " C = sin(A)", cC(i,j) = sin(cA(i,j)), C= sin(A))
  ADD_TEST_2D_EW(PFX + " C = sinh(A)", cC(i,j) = sinh(cA(i,j)), C= sinh(A))
  ADD_TEST_2D_EW(PFX + " C = sqrt(A)", cC(i,j) = sqrt(cA(i,j)), C= sqrt(A))
  ADD_TEST_2D_EW(PFX + " C = tan(A)", cC(i,j) = tan(cA(i,j)), C= tan(A))
  ADD_TEST_2D_EW(PFX + " C = tanh(A)", cC(i,j) = tanh(cA(i,j)), C= tanh(A))

  ADD_TEST_2D_EW(PFX + " C = A.*B", cC(i,j) = cA(i,j)*cB(i,j), C= A*B)
  ADD_TEST_2D_EW(PFX + " C = A./B", cC(i,j) = cA(i,j)/cB(i,j), C= A/B)
  ADD_TEST_2D_EW(PFX + " C = pow(A,B)", cC(i,j) = pow(cA(i,j), cB(i,j)), C= pow(A,B))

  ADD_TEST_2D_EW(PFX + " C = eye(M, N)", cC(i,j) = i==j, C= eye(M, N, C.dtype(), C.context()))
  ADD_TEST_2D_EW(PFX + " C = outer(x, y)", cC(i,j) = cx[i]*cy[j], C= outer(x,y))

  ADD_TEST_2D_EW(PFX + " C = A==B", cC(i,j) = cA(i,j)==cB(i,j), C= cast(A==B, dtype))
  ADD_TEST_2D_EW(PFX + " C = A>=B", cC(i,j) = cA(i,j)>=cB(i,j), C= cast(A>=B, dtype))
  ADD_TEST_2D_EW(PFX + " C = A>B", cC(i,j) = cA(i,j)>cB(i,j), C= cast(A>B, dtype))
  ADD_TEST_2D_EW(PFX + " C = A<=B", cC(i,j) = cA(i,j)<=cB(i,j), C= cast(A<=B, dtype))
  ADD_TEST_2D_EW(PFX + " C = A<B", cC(i,j) = cA(i,j)<cB(i,j), C= cast(A<B, dtype))
  ADD_TEST_2D_EW(PFX + " C = A!=B", cC(i,j) = cA(i,j)!=cB(i,j), C= cast(A!=B, dtype))

}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  int_t M = 173;
  int_t N = 241;
  int_t SUBM = 7;
  int_t SUBN = 11;

  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cA, A, ctx);
  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cB, B, ctx);
  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cC, C, ctx);
  INIT_VECTOR(M, SUBM, 5, 3, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 7, 2, cy, y, ctx);

  test_impl("FULL", cA, cB, cC, cx, cy, A, B, C, x, y, nfail, npass);
  test_impl("SUB", cA_s, cB_s, cC_s, cx_s, cy_s, A_s, B_s, C_s, x_s, y_s, nfail, npass);
}

int main()
{
  return run_test(test<float>, test<double>);
}
