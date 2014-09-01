#include "common.hpp"
#include <cmath>
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

#include "atidlas/templates/reduction.hpp"
#include "atidlas/execute.hpp"

template<typename NumericT, class XType, class YType>
void test_reduction(NumericT epsilon,  XType & cx, YType & cy)
{
  using namespace viennacl::linalg;
  using namespace std;

  atidlas::reduction_parameters parameters(1, 32, 128, atidlas::FETCH_FROM_GLOBAL_CONTIGUOUS);
  int failure_count = 0;
  NumericT cs = 0;
  NumericT tmp = 0;
  viennacl::scalar<NumericT> s(0);

  viennacl::vector<NumericT> xtmp(cx.internal_size());
  typename vector_maker<XType>::result_type x = vector_maker<XType>::make(xtmp, cx);
  viennacl::vector<NumericT> ytmp(cy.internal_size());
  typename vector_maker<YType>::result_type y = vector_maker<YType>::make(ytmp, cy);


#define RUN_TEST(NAME, CPU_REDUCTION, ASSIGNMENT, GPU_STATEMENT) \
  cout << NAME "..." << flush;\
  cs = 0;\
  for(int_t i = 0 ; i < cx.size() ; ++i)\
    CPU_REDUCTION;\
  cs= ASSIGNMENT ;\
  atidlas::execute(atidlas::reduction_template(parameters),\
                   GPU_STATEMENT,\
                   viennacl::ocl::current_context(), true);\
  tmp = s;\
  if((std::abs(cs - tmp)/std::max(cs, tmp)) > epsilon)\
  {\
    failure_count++;\
    cout << " [Failure!]" << endl;\
  }\
  else\
    cout << endl;

  RUN_TEST("s = x.y", cs+=cx[i]*cy[i], cs, viennacl::scheduler::statement(s, viennacl::op_assign(), viennacl::linalg::inner_prod(x, y)))

#undef RUN_TEST

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename NumericT>
void test_impl(NumericT epsilon)
{
  int_t N = 24378;
  INIT_VECTOR_AND_PROXIES(N, 4, 5, x);
  INIT_VECTOR_AND_PROXIES(N, 7, 8, y);

#define TEST_OPERATIONS(XTYPE, YTYPE)\
  std::cout << "> x : " #XTYPE " | y : " #YTYPE  << std::endl;\
  test_reduction(epsilon, x_ ## XTYPE, y_ ## YTYPE);\

  TEST_OPERATIONS(vector, vector)
  TEST_OPERATIONS(vector, range)
  TEST_OPERATIONS(vector, slice)
  TEST_OPERATIONS(range, vector)
  TEST_OPERATIONS(range, range)
  TEST_OPERATIONS(range, slice)
  TEST_OPERATIONS(slice, vector)
  TEST_OPERATIONS(slice, range)
  TEST_OPERATIONS(slice, slice)
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
