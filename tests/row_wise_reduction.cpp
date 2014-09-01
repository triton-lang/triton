//#define VIENNACL_DEBUG_ALL
#include <cmath>
#include "common.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "atidlas/templates/row_wise_reduction.hpp"
#include "atidlas/execute.hpp"

template<typename NumericT, class AType, class XType, class YType>
void test_row_wise_reduction(NumericT epsilon, YType & cy, AType & cA, AType & cAT, XType & cx)
{
  using viennacl::trans;

  atidlas::row_wise_reduction_parameters parameters(4, 8, 8, 128, atidlas::FETCH_FROM_GLOBAL_CONTIGUOUS);
  int failure_count = 0;

//  viennacl::matrix<NumericT, viennacl::row_major> ArowTmp(cA.internal_size1(), cA.internal_size2());
//  viennacl::matrix<NumericT, viennacl::row_major> ATrowTmp(cAT.internal_size1(), cAT.internal_size2());
  viennacl::matrix<NumericT, viennacl::column_major> AcolTmp(cA.internal_size1(), cA.internal_size2());
  viennacl::matrix<NumericT, viennacl::column_major> ATcolTmp(cAT.internal_size1(), cAT.internal_size2());
  viennacl::vector<NumericT> xtmp(cx.internal_size());
  viennacl::vector<NumericT> ytmp(cy.internal_size());

//  typename matrix_maker<AType, viennacl::row_major>::result_type Arow = matrix_maker<AType, viennacl::row_major>::make(ArowTmp, cA);
//  typename matrix_maker<AType, viennacl::row_major>::result_type ATrow = matrix_maker<AType, viennacl::row_major>::make(ATrowTmp, cAT);
  typename matrix_maker<AType, viennacl::column_major>::result_type Acol = matrix_maker<AType, viennacl::column_major>::make(AcolTmp, cA);
  typename matrix_maker<AType, viennacl::column_major>::result_type ATcol = matrix_maker<AType, viennacl::column_major>::make(ATcolTmp, cAT);
  typename vector_maker<XType>::result_type x = vector_maker<XType>::make(xtmp, cx);
  typename vector_maker<YType>::result_type y = vector_maker<YType>::make(ytmp, cy);

  simple_vector<NumericT> ground(cA.size1());
  simple_vector<NumericT> buffer(cA.size1());

#define TEST_OPERATION(NAME, REDUCTION, ASSIGNMENT, TRANS_FLAG, GPU_STATEMENT)\
  std::cout << NAME "..." << std::flush;\
  for(int i = 0 ; i < ground.size() ; ++i)\
  {\
    NumericT yi = 0;\
    for(int j = 0 ; j < cx.size() ; ++j)\
      yi+=REDUCTION;\
    ground[i] = ASSIGNMENT;\
  }\
  atidlas::execute(atidlas::row_wise_reduction_template(parameters,TRANS_FLAG),\
                   GPU_STATEMENT,\
                   viennacl::ocl::current_context(), true);\
  viennacl::copy(y, buffer);\
  if(failure_vector(ground, buffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;

//  std::cout << "> row" << std::endl;
//  TEST_OPERATION("y = A.x", cA(i,j)*cx[j], yi, 'N', viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(Arow, x)));
//  TEST_OPERATION("y = A'.x", cA(i,j)*cx[j], yi, 'T', viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(trans(ATrow), x)));
  std::cout << "> col" << std::endl;
  TEST_OPERATION("y = A.x", cA(i,j)*cx[j], yi, 'N', viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(Acol, x)));
  TEST_OPERATION("y = A'.x", cAT(j,i)*cx[j], yi, 'T', viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(trans(ATcol), x)));

  if(failure_count>0)
    exit(EXIT_FAILURE);
}

template<typename NumericT>
void test_impl(NumericT epsilon)
{
  int_t M = 328;
  int_t N = 391;
  INIT_VECTOR_AND_PROXIES(M, 25, 8, y);
  INIT_MATRIX_AND_PROXIES(M, 33, 4, N, 51, 2, A);
  INIT_VECTOR_AND_PROXIES(N, 14, 5, x);


#define TEST_OPERATIONS(YTYPE, ATYPE, XTYPE)\
  std::cout << ">> y : " #YTYPE " | A : " #ATYPE " | x : " #XTYPE << std::endl;\
  test_row_wise_reduction(epsilon, y_ ## YTYPE, A_ ## ATYPE, AT_ ## ATYPE, x_ ## XTYPE);\

  TEST_OPERATIONS(vector, matrix, vector)
  TEST_OPERATIONS(vector, matrix, range)
  TEST_OPERATIONS(vector, matrix, slice)
  TEST_OPERATIONS(vector, range, vector)
  TEST_OPERATIONS(vector, range, range)
  TEST_OPERATIONS(vector, range, slice)
  TEST_OPERATIONS(vector, slice, vector)
  TEST_OPERATIONS(vector, slice, range)
  TEST_OPERATIONS(vector, slice, slice)

  TEST_OPERATIONS(range, matrix, vector)
  TEST_OPERATIONS(range, matrix, range)
  TEST_OPERATIONS(range, matrix, slice)
  TEST_OPERATIONS(range, range, vector)
  TEST_OPERATIONS(range, range, range)
  TEST_OPERATIONS(range, range, slice)
  TEST_OPERATIONS(range, slice, vector)
  TEST_OPERATIONS(range, slice, range)
  TEST_OPERATIONS(range, slice, slice)

  TEST_OPERATIONS(slice, matrix, vector)
  TEST_OPERATIONS(slice, matrix, range)
  TEST_OPERATIONS(slice, matrix, slice)
  TEST_OPERATIONS(slice, range, vector)
  TEST_OPERATIONS(slice, range, range)
  TEST_OPERATIONS(slice, range, slice)
  TEST_OPERATIONS(slice, slice, vector)
  TEST_OPERATIONS(slice, slice, range)
  TEST_OPERATIONS(slice, slice, slice)
}

int main()
{
  std::cout << ">>> float" << std::endl;
  test_impl<float>(1e-4);
  std::cout << ">>> double" << std::endl;
  test_impl<double>(1e-9);
  std::cout << "---" << std::endl;
  std::cout << "Passed" << std::endl;

  return EXIT_SUCCESS;
}
