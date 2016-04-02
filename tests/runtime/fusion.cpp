#include "isaac/symbolic/execute.h"
#include "isaac/symbolic/expression/io.h"
#include "isaac/array.h"

namespace sc = isaac;

int main()
{
  int nfail = 0, npass = 0;
  sc::array A(3,4), B(3, 4), C(3,3);
  sc::array y(3), u(4), x(4), z(7);
  sc::scalar da(0);

  #define ADD_TMP_TEST(NAME, RESULT_TYPE, NTMP, SCEXPR) \
  {\
    std::cout << NAME << "...";\
    sc::expression_tree tree = SCEXPR;\
    sc::symbolic::detail::breakpoints_t breakpoints;\
    sc::expression_type type = sc::symbolic::detail::parse(tree, breakpoints);\
    if(!(type == RESULT_TYPE && breakpoints.size()==NTMP)){\
      std::cout << " [Failure!]" << std::endl;\
      nfail++;\
    }\
    else{\
      std::cout << std::endl;\
      npass++;\
    }\
  }

  /* Fully fused */
  //Elementwise 1D
  ADD_TMP_TEST("y = x", sc::ELEMENTWISE_1D, 0, sc::assign(u, x))
  ADD_TMP_TEST("y = ax + by", sc::ELEMENTWISE_1D, 0, sc::assign(u, 2*x + 3*u))
  ADD_TMP_TEST("y = ax + b(y + exp(x))", sc::ELEMENTWISE_1D, 0, sc::assign(u, 2*x + 3*(u + sc::exp(x))))

  //Elementwise 2D
  ADD_TMP_TEST("B = A", sc::ELEMENTWISE_2D, 0, sc::assign(B, A))
  ADD_TMP_TEST("B = aA + bB", sc::ELEMENTWISE_2D, 0, sc::assign(B, 2*A + 3*B))
  ADD_TMP_TEST("B = aA + b(B + exp(A))", sc::ELEMENTWISE_2D, 0, sc::assign(B, 2*A + 3*(B + sc::exp(A))))

  //1D reduction
  ADD_TMP_TEST("da = sum(x)", sc::REDUCE_1D, 0, sc::assign(da, sum(x)));
  ADD_TMP_TEST("da = sum(ax + by)", sc::REDUCE_1D, 0, sc::assign(da, sum(2*x + 3*u)));

  //2D reduction
  ADD_TMP_TEST("x = sum(A, 0)", sc::REDUCE_2D_COLS, 0, sc::assign(x, sum(A, 0)));
  ADD_TMP_TEST("x = sum(aA + bB, 0)", sc::REDUCE_2D_COLS, 0, sc::assign(x, sum(2*A + 3*B, 0)));
  ADD_TMP_TEST("y = sum(A, 1)", sc::REDUCE_2D_ROWS, 0, sc::assign(y, sum(A, 1)));
  ADD_TMP_TEST("y = sum(aA + bB, 1)", sc::REDUCE_2D_ROWS, 0, sc::assign(y, sum(2*A + 3*B, 1)));

  //Dot
  ADD_TMP_TEST("y = dot(A,x)", sc::REDUCE_2D_ROWS, 0, sc::assign(y, dot(A,x)));
  ADD_TMP_TEST("y = dot(A + B, x)", sc::REDUCE_2D_ROWS, 0, sc::assign(y, dot(A + B, x)));
  ADD_TMP_TEST("y = dot(A + B, x + z)", sc::REDUCE_2D_ROWS, 0, sc::assign(y, dot(A + B, x + u)));

  /* Partially fused */
  ADD_TMP_TEST("da = sum(ax + by) + sum(z)", sc::ELEMENTWISE_1D, 2, sc::assign(da, sum(2*x + 3*u) + sum(z)));
  ADD_TMP_TEST("x = sum(ax + by)*sum(aA + bB, 0)", sc::ELEMENTWISE_1D, 2, sc::assign(da, sum(2*x + 3*u) + sum(z)));

}
