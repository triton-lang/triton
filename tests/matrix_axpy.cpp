#include "common.hpp"
#include <cmath>
#include "viennacl/matrix.hpp"

#include "atidlas/templates/matrix_axpy.hpp"
#include "atidlas/execute.hpp"

template<typename NumericT, class AType, class BType, class CType>
void test_element_wise_matrix(NumericT epsilon,  AType & cA, BType & cB, CType & cC)
{
  using namespace viennacl::linalg;
  using namespace std;

  atidlas::matrix_axpy_parameters_type parameters(1, 8, 8, 32, 32, atidlas::FETCH_FROM_GLOBAL_STRIDED);
  int failure_count = 0;
  CType tmp = cC;

  NumericT a = 3.12, b = 3.5;
  viennacl::scalar<NumericT> da(a), db(b);

  viennacl::matrix<NumericT, viennacl::column_major> Atmp(cA.internal_size1(), cA.internal_size2());
  typename matrix_maker<AType, viennacl::column_major>::result_type A = matrix_maker<AType, viennacl::column_major>::make(Atmp, cA);
  viennacl::matrix<NumericT, viennacl::column_major> Btmp(cB.internal_size1(), cB.internal_size2());
  typename matrix_maker<BType, viennacl::column_major>::result_type B = matrix_maker<BType, viennacl::column_major>::make(Btmp, cB);
  viennacl::matrix<NumericT, viennacl::column_major> Ctmp(cC.internal_size1(), cC.internal_size2());
  typename matrix_maker<CType, viennacl::column_major>::result_type C = matrix_maker<CType, viennacl::column_major>::make(Ctmp, cC);


#define RUN_TEST(NAME, CPU_LOOP, GPU_STATEMENT) \
  cout << NAME "..." << flush;\
  for(int_t i = 0 ; i < cC.size1() ; ++i)\
    for(int_t j = 0 ; j < cC.size2() ; ++j)\
      CPU_LOOP;\
  atidlas::execute(atidlas::matrix_axpy_template(parameters),\
                   GPU_STATEMENT,\
                   viennacl::ocl::current_context(), true);\
  viennacl::copy(C, tmp);\
  if(failure_matrix(cC, tmp, epsilon))\
  {\
    failure_count++;\
    cout << " [Failure!]" << endl;\
  }\
  else\
    cout << endl;

  RUN_TEST("C = A", cC(i,j) = cA(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), A))
  RUN_TEST("C = A + B", cC(i,j) = cA(i,j) + cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), A + B))
  RUN_TEST("C = A - B", cC(i,j) = cA(i,j) - cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), A - B))
  RUN_TEST("C = A + B + C", cC(i,j) = cA(i,j) + cB(i,j) + cC(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), A + B + C))

  RUN_TEST("C = a*A", cC(i,j) = a*cA(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), a*A))
  RUN_TEST("C = da*A", cC(i,j) = a*cA(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), da*A))
  RUN_TEST("C = a*A + b*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), a*A + b*B))
  RUN_TEST("C = da*A + b*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), da*A + b*B))
  RUN_TEST("C = a*A + db*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), a*A + db*B))
  RUN_TEST("C = da*A + db*B", cC(i,j) = a*cA(i,j) + b*cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), da*A + db*B))

//  RUN_TEST("C = abs(A)", cC(i,j) = abs(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_abs(A)))
  RUN_TEST("C = acos(A)", cC(i,j) = acos(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_acos(A)))
  RUN_TEST("C = asin(A)", cC(i,j) = asin(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_asin(A)))
  RUN_TEST("C = atan(A)", cC(i,j) = atan(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_atan(A)))
  RUN_TEST("C = ceil(A)", cC(i,j) = ceil(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_ceil(A)))
  RUN_TEST("C = cos(A)", cC(i,j) = cos(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_cos(A)))
  RUN_TEST("C = cosh(A)", cC(i,j) = cosh(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_cosh(A)))
  RUN_TEST("C = exp(A)", cC(i,j) = exp(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_exp(A)))
  RUN_TEST("C = fabs(A)", cC(i,j) = fabs(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_fabs(A)))
  RUN_TEST("C = floor(A)", cC(i,j) = floor(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_floor(A)))
  RUN_TEST("C = log(A)", cC(i,j) = log(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_log(A)))
  RUN_TEST("C = log10(A)", cC(i,j) = log10(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_log10(A)))
  RUN_TEST("C = sin(A)", cC(i,j) = sin(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_sin(A)))
  RUN_TEST("C = sinh(A)", cC(i,j) = sinh(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_sinh(A)))
  RUN_TEST("C = sqrt(A)", cC(i,j) = sqrt(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_sqrt(A)))
  RUN_TEST("C = tan(A)", cC(i,j) = tan(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_tan(A)))
  RUN_TEST("C = tanh(A)", cC(i,j) = tanh(cA(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_tanh(A)))

  RUN_TEST("C = A./B", cC(i,j) = cA(i,j)/cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_div(A,B)))
  RUN_TEST("C = A==B", cC(i,j) = cA(i,j)==cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_eq(A,B)))
  RUN_TEST("C = A>=B", cC(i,j) = cA(i,j)>=cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_geq(A,B)))
  RUN_TEST("C = A>B", cC(i,j) = cA(i,j)>cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_greater(A,B)))
  RUN_TEST("C = A<=B", cC(i,j) = cA(i,j)<=cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_leq(A,B)))
  RUN_TEST("C = A<B", cC(i,j) = cA(i,j)<cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_less(A,B)))
  RUN_TEST("C = A!=B", cC(i,j) = cA(i,j)!=cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_neq(A,B)))
  RUN_TEST("C = pow(A,B)", cC(i,j) = pow(cA(i,j), cB(i,j)), viennacl::scheduler::statement(C, viennacl::op_assign(), element_pow(A,B)))
  RUN_TEST("C = A.*B", cC(i,j) = cA(i,j)*cB(i,j), viennacl::scheduler::statement(C, viennacl::op_assign(), element_prod(A,B)))

#undef RUN_TEST

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename NumericT>
void test_impl(NumericT epsilon)
{
  int_t M = 241;
  int_t N = 278;
  INIT_MATRIX_AND_PROXIES(M, 4, 5, N, 7, 6, C);
  INIT_MATRIX_AND_PROXIES(M, 2, 4, N, 5, 8, A);
  INIT_MATRIX_AND_PROXIES(M, 2, 3, N, 3, 4, B);

#define TEST_OPERATIONS(CTYPE, ATYPE, BTYPE)\
  std::cout << "> C : " #CTYPE " | A : " #ATYPE " | B : " #BTYPE << std::endl;\
  test_element_wise_matrix(epsilon, A_ ## ATYPE, B_ ## BTYPE, C_ ## CTYPE);\

  TEST_OPERATIONS(matrix, matrix, matrix)
  TEST_OPERATIONS(matrix, matrix, range)
  TEST_OPERATIONS(matrix, matrix, slice)
  TEST_OPERATIONS(matrix, range, matrix)
  TEST_OPERATIONS(matrix, range, range)
  TEST_OPERATIONS(matrix, range, slice)
  TEST_OPERATIONS(matrix, slice, matrix)
  TEST_OPERATIONS(matrix, slice, range)
  TEST_OPERATIONS(matrix, slice, slice)

  TEST_OPERATIONS(range, matrix, matrix)
  TEST_OPERATIONS(range, matrix, range)
  TEST_OPERATIONS(range, matrix, slice)
  TEST_OPERATIONS(range, range, matrix)
  TEST_OPERATIONS(range, range, range)
  TEST_OPERATIONS(range, range, slice)
  TEST_OPERATIONS(range, slice, matrix)
  TEST_OPERATIONS(range, slice, range)
  TEST_OPERATIONS(range, slice, slice)

  TEST_OPERATIONS(slice, matrix, matrix)
  TEST_OPERATIONS(slice, matrix, range)
  TEST_OPERATIONS(slice, matrix, slice)
  TEST_OPERATIONS(slice, range, matrix)
  TEST_OPERATIONS(slice, range, range)
  TEST_OPERATIONS(slice, range, slice)
  TEST_OPERATIONS(slice, slice, matrix)
  TEST_OPERATIONS(slice, slice, range)
  TEST_OPERATIONS(slice, slice, slice)
}

int main()
{
  std::cout << ">> float" << std::endl;
  test_impl<float>(1e-4);
  std::cout << ">> double" << std::endl;
  test_impl<double>(1e-9);
  std::cout << "---" << std::endl;
  std::cout << "Passed" << std::endl;

  return EXIT_SUCCESS;
}
