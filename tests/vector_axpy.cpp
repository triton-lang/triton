
#include "common.hpp"
#include <cmath>
#include "viennacl/vector.hpp"

#include "atidlas/backend/templates/vector_axpy.hpp"
#include "atidlas/model/model.hpp"

template<typename NumericT, class XType, class YType, class ZType>
void test_element_wise_vector(NumericT epsilon,  XType & cx, YType & cy, ZType & cz)
{
  using namespace viennacl::linalg;
  using namespace std;

  atidlas::vector_axpy_parameters parameters(4, 32, 128, atidlas::FETCH_FROM_GLOBAL_CONTIGUOUS);

  int failure_count = 0;
  ZType buffer = cz;

  NumericT a = 3.12, b = 3.5;
  viennacl::scalar<NumericT> da(a), db(b);

  viennacl::vector<NumericT> xtmp(cx.internal_size());
  typename vector_maker<XType>::result_type x = vector_maker<XType>::make(xtmp, cx);
  viennacl::vector<NumericT> ytmp(cy.internal_size());
  typename vector_maker<YType>::result_type y = vector_maker<YType>::make(ytmp, cy);
  viennacl::vector<NumericT> ztmp(cz.internal_size());
  typename vector_maker<ZType>::result_type z = vector_maker<ZType>::make(ztmp, cz);


#define RUN_TEST_VECTOR_AXPY(NAME, CPU_LOOP, GPU_STATEMENT) \
  {\
  atidlas::model model(atidlas::vector_axpy_template(parameters), viennacl::ocl::current_context(), viennacl::ocl::current_device());\
  cout << NAME "..." << flush;\
  for(int_t i = 0 ; i < cz.size() ; ++i)\
    CPU_LOOP;\
  model.execute(GPU_STATEMENT);\
  viennacl::copy(z, buffer);\
  if(failure_vector(cz, buffer, epsilon))\
  {\
    failure_count++;\
    cout << " [Failure!]" << endl;\
  }\
  else\
    cout << endl;\
  }

  RUN_TEST_VECTOR_AXPY("z = x", cz[i] = cx[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x))
  RUN_TEST_VECTOR_AXPY("z = x + y", cz[i] = cx[i] + cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x + y))
  RUN_TEST_VECTOR_AXPY("z = x - y", cz[i] = cx[i] - cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x - y))
  RUN_TEST_VECTOR_AXPY("z = x + y + z", cz[i] = cx[i] + cy[i] + cz[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x + y + z))

  RUN_TEST_VECTOR_AXPY("z = a*x", cz[i] = a*cx[i], viennacl::scheduler::statement(z, viennacl::op_assign(), a*x))
  RUN_TEST_VECTOR_AXPY("z = da*x", cz[i] = a*cx[i], viennacl::scheduler::statement(z, viennacl::op_assign(), da*x))
  RUN_TEST_VECTOR_AXPY("z = a*x + b*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), a*x + b*y))
  RUN_TEST_VECTOR_AXPY("z = da*x + b*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), da*x + b*y))
  RUN_TEST_VECTOR_AXPY("z = a*x + db*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), a*x + db*y))
  RUN_TEST_VECTOR_AXPY("z = da*x + db*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), da*x + db*y))

  RUN_TEST_VECTOR_AXPY("z = exp(x)", cz[i] = exp(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_exp(x)))
//  RUN_TEST_VECTOR_AXPY("z = abs(x)", cz[i] = abs(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_abs(x)))
  RUN_TEST_VECTOR_AXPY("z = acos(x)", cz[i] = acos(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_acos(x)))
  RUN_TEST_VECTOR_AXPY("z = asin(x)", cz[i] = asin(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_asin(x)))
  RUN_TEST_VECTOR_AXPY("z = atan(x)", cz[i] = atan(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_atan(x)))
  RUN_TEST_VECTOR_AXPY("z = ceil(x)", cz[i] = ceil(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_ceil(x)))
  RUN_TEST_VECTOR_AXPY("z = cos(x)", cz[i] = cos(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_cos(x)))
  RUN_TEST_VECTOR_AXPY("z = cosh(x)", cz[i] = cosh(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_cosh(x)))
  RUN_TEST_VECTOR_AXPY("z = exp(x)", cz[i] = exp(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_exp(x)))
  RUN_TEST_VECTOR_AXPY("z = fabs(x)", cz[i] = fabs(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_fabs(x)))
  RUN_TEST_VECTOR_AXPY("z = floor(x)", cz[i] = floor(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_floor(x)))
  RUN_TEST_VECTOR_AXPY("z = log(x)", cz[i] = log(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_log(x)))
  RUN_TEST_VECTOR_AXPY("z = log10(x)", cz[i] = log10(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_log10(x)))
  RUN_TEST_VECTOR_AXPY("z = sin(x)", cz[i] = sin(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_sin(x)))
  RUN_TEST_VECTOR_AXPY("z = sinh(x)", cz[i] = sinh(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_sinh(x)))
  RUN_TEST_VECTOR_AXPY("z = sqrt(x)", cz[i] = sqrt(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_sqrt(x)))
  RUN_TEST_VECTOR_AXPY("z = tan(x)", cz[i] = tan(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_tan(x)))
  RUN_TEST_VECTOR_AXPY("z = tanh(x)", cz[i] = tanh(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_tanh(x)))

  RUN_TEST_VECTOR_AXPY("z = x./y", cz[i] = cx[i]/cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_div(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x==y", cz[i] = cx[i]==cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_eq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x>=y", cz[i] = cx[i]>=cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_geq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x>y", cz[i] = cx[i]>cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_greater(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x<=y", cz[i] = cx[i]<=cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_leq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x<y", cz[i] = cx[i]<cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_less(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x!=y", cz[i] = cx[i]!=cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_neq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = pow(x,y)", cz[i] = pow(cx[i], cy[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_pow(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x.*y", cz[i] = cx[i]*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_prod(x,y)))

#undef RUN_TEST_VECTOR_AXPY

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename NumericT>
void test_impl(NumericT epsilon)
{
  int_t N = 24378;
  INIT_VECTOR_AND_PROXIES(N, 4, 5, x);
  INIT_VECTOR_AND_PROXIES(N, 7, 8, y);
  INIT_VECTOR_AND_PROXIES(N, 15, 12, z);

#define TEST_OPERATIONS(XTYPE, YTYPE, ZTYPE)\
  std::cout << "> x : " #XTYPE " | y : " #YTYPE " | z : " #ZTYPE << std::endl;\
  test_element_wise_vector(epsilon, x_ ## XTYPE, y_ ## YTYPE, z_ ## ZTYPE);\

  TEST_OPERATIONS(vector, vector, vector)
  TEST_OPERATIONS(vector, vector, range)
  TEST_OPERATIONS(vector, vector, slice)
  TEST_OPERATIONS(vector, range, vector)
  TEST_OPERATIONS(vector, range, range)
  TEST_OPERATIONS(vector, range, slice)
  TEST_OPERATIONS(vector, slice, vector)
  TEST_OPERATIONS(vector, slice, range)
  TEST_OPERATIONS(vector, slice, slice)

  TEST_OPERATIONS(range, vector, vector)
  TEST_OPERATIONS(range, vector, range)
  TEST_OPERATIONS(range, vector, slice)
  TEST_OPERATIONS(range, range, vector)
  TEST_OPERATIONS(range, range, range)
  TEST_OPERATIONS(range, range, slice)
  TEST_OPERATIONS(range, slice, vector)
  TEST_OPERATIONS(range, slice, range)
  TEST_OPERATIONS(range, slice, slice)

  TEST_OPERATIONS(slice, vector, vector)
  TEST_OPERATIONS(slice, vector, range)
  TEST_OPERATIONS(slice, vector, slice)
  TEST_OPERATIONS(slice, range, vector)
  TEST_OPERATIONS(slice, range, range)
  TEST_OPERATIONS(slice, range, slice)
  TEST_OPERATIONS(slice, slice, vector)
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
