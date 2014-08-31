#include <cmath>
#include "common.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "atidlas/templates/row_wise_reduction_template.hpp"
#include "atidlas/execute.hpp"

template<typename NumericT, class AType, class XType, class YType>
void test_row_wise_reduction(NumericT epsilon, atidlas::row_wise_reduction_parameters const & parameters,
                  YType & cy, AType & cA, AType & cAT, XType & cx)
{
  int failure_count = 0;

  viennacl::matrix<NumericT, viennacl::row_major> ArowTmp(cA.internal_size1(), cA.internal_size2());
  viennacl::matrix<NumericT, viennacl::row_major> ATrowTmp(cAT.internal_size1(), cAT.internal_size2());
  viennacl::matrix<NumericT, viennacl::column_major> AcolTmp(cA.internal_size1(), cA.internal_size2());
  viennacl::matrix<NumericT, viennacl::column_major> ATcolTmp(cAT.internal_size1(), cAT.internal_size2());
  viennacl::vector<NumericT> xtmp(cx.internal_size());
  viennacl::vector<NumericT> ytmp(cy.internal_size());

  typename matrix_maker<AType, viennacl::row_major>::result_type Arow = matrix_maker<AType, viennacl::row_major>::make(ArowTmp, cA);
  typename matrix_maker<AType, viennacl::row_major>::result_type ATrow = matrix_maker<AType, viennacl::row_major>::make(ATrowTmp, cAT);
  typename matrix_maker<AType, viennacl::column_major>::result_type Acol = matrix_maker<AType, viennacl::column_major>::make(AcolTmp, cA);
  typename matrix_maker<AType, viennacl::column_major>::result_type ATcol = matrix_maker<AType, viennacl::column_major>::make(ATcolTmp, cAT);
  typename vector_maker<XType>::result_type x = vector_maker<XType>::make(xtmp, cx);
  typename vector_maker<XType>::result_type y = vector_maker<XType>::make(ytmp, cy);

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

  TEST_OPERATION("y = A.x", cA(i,j)*cx[j], yi, 'N', viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(Arow, x)));

  if(failure_count>0)
    exit(EXIT_FAILURE);
}

template<typename NumericT>
void test_impl(NumericT epsilon)
{
  atidlas::row_wise_reduction_parameters parameters(4, 8, 8, 128, atidlas::FETCH_FROM_GLOBAL_CONTIGUOUS);

  int_t M = 329;
  int_t N = 391;
  int x_start = 145, y_start = 93, M_start = 243, N_start = 351;
  int x_stride = 5, y_stride = 8, M_stride = 3, N_stride = 7;
  viennacl::range xr(x_start, N + x_start), yr(y_start, M + y_start), Mr(M_start, M + M_start), Nr(N_start, N + N_start);
  viennacl::slice xs(x_start, x_stride, N), ys(y_start, y_stride, M), Ms(M_start, M_stride, M), Ns(N_start, N_stride, N);


  simple_vector<NumericT> x_vector(N);
  simple_vector<NumericT> x_range_holder(N + x_start);
  simple_vector<NumericT> x_slice_holder(x_start + N*x_stride);
  init_rand(x_vector);
  init_rand(x_range_holder);
  init_rand(x_slice_holder);
  simple_vector_range< simple_vector<NumericT> > x_range(x_range_holder, xr);
  simple_vector_slice< simple_vector<NumericT> > x_slice(x_slice_holder, xs);

  simple_vector<NumericT> y_vector(N);
  simple_vector<NumericT> y_range_holder(N + y_start);
  simple_vector<NumericT> y_slice_holder(y_start + N*y_stride);
  init_rand(y_vector);
  init_rand(y_range_holder);
  init_rand(y_slice_holder);
  simple_vector_range< simple_vector<NumericT> > y_range(y_range_holder, xr);
  simple_vector_slice< simple_vector<NumericT> > y_slice(y_slice_holder, xs);

  simple_matrix<NumericT> A_matrix(M, N);
  simple_matrix<NumericT> A_range_holder(M_start + M, N_start + N);
  simple_matrix<NumericT> A_slice_holder(M_start + M*M_stride, N_start + N*N_stride);
  init_rand(A_matrix);
  init_rand(A_range_holder);
  init_rand(A_slice_holder);
  simple_matrix_range< simple_matrix<NumericT> > A_range(A_range_holder, Mr, Nr);
  simple_matrix_slice< simple_matrix<NumericT> > A_slice(A_slice_holder, Ms, Ns);

  simple_matrix<NumericT> AT_matrix = simple_trans(A_matrix);
  simple_matrix<NumericT> AT_range_holder = simple_trans(A_range_holder);
  simple_matrix<NumericT> AT_slice_holder = simple_trans(A_slice_holder);
  simple_matrix_range< simple_matrix<NumericT> > AT_range(AT_range_holder, Nr, Mr);
  simple_matrix_slice< simple_matrix<NumericT> > AT_slice(AT_slice_holder, Ns, Ms);


#define TEST_OPERATIONS(YTYPE, ATYPE, XTYPE)\
  std::cout << "> y : " #YTYPE " | A : " #ATYPE " | x : " #XTYPE << std::endl;\
  test_row_wise_reduction(epsilon, parameters, y_ ## YTYPE, A_ ## ATYPE, AT_ ## ATYPE, x_ ## XTYPE);\

  TEST_OPERATIONS(vector, matrix, vector)
//  TEST_OPERATIONS(vector, vector, range)
//  TEST_OPERATIONS(vector, vector, slice)
//  TEST_OPERATIONS(vector, range, vector)
//  TEST_OPERATIONS(vector, range, range)
//  TEST_OPERATIONS(vector, range, slice)
//  TEST_OPERATIONS(vector, slice, vector)
//  TEST_OPERATIONS(vector, slice, range)
//  TEST_OPERATIONS(vector, slice, slice)

//  TEST_OPERATIONS(range, vector, vector)
//  TEST_OPERATIONS(range, vector, range)
//  TEST_OPERATIONS(range, vector, slice)
//  TEST_OPERATIONS(range, range, vector)
//  TEST_OPERATIONS(range, range, range)
//  TEST_OPERATIONS(range, range, slice)
//  TEST_OPERATIONS(range, slice, vector)
//  TEST_OPERATIONS(range, slice, range)
//  TEST_OPERATIONS(range, slice, slice)

//  TEST_OPERATIONS(slice, vector, vector)
//  TEST_OPERATIONS(slice, vector, range)
//  TEST_OPERATIONS(slice, vector, slice)
//  TEST_OPERATIONS(slice, range, vector)
//  TEST_OPERATIONS(slice, range, range)
//  TEST_OPERATIONS(slice, range, slice)
//  TEST_OPERATIONS(slice, slice, vector)
//  TEST_OPERATIONS(slice, slice, range)
//  TEST_OPERATIONS(slice, slice, slice)
}

int main()
{
  std::cout << ">> float" << std::endl;
  test_impl<float>(1e-5);
  std::cout << ">> double" << std::endl;
  test_impl<double>(1e-9);
  std::cout << "---" << std::endl;
  std::cout << "Passed" << std::endl;

  return EXIT_SUCCESS;
}
