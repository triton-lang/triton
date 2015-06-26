#include <cmath>
#include <iostream>

#include "common.hpp"
#include "isaac/array.h"
#include "isaac/wrap/clBLAS.h"

namespace ad = isaac;
typedef ad::int_t int_t;

template<typename T>
void test_reduction(T epsilon,  simple_vector_base<T> & cx, simple_vector_base<T> & cy,
                                ad::array & x, ad::array & y)
{
  using namespace std;
  ad::driver::Context const & ctx = x.context();
  int_t N = cx.size();
  ad::driver::CommandQueue queue = ad::driver::queues[ctx][0];
  cl_command_queue clqueue = (*queue.handle().cl)();

  unsigned int failure_count = 0;

  isaac::numeric_type dtype = ad::to_numeric_type<T>::value;

  T cs = 0;
  T tmp = 0;
  isaac::scalar ds(dtype, ctx);

#define RUN_TEST(NAME, CPU_REDUCTION, INIT, ASSIGNMENT, GPU_REDUCTION) \
  cout << PREFIX << " " << NAME "..." << flush;\
  cs = INIT;\
  for(int_t i = 0 ; i < N ; ++i)\
    CPU_REDUCTION;\
  cs= ASSIGNMENT ;\
  GPU_REDUCTION;\
  tmp = ds;\
  if((std::abs(cs - tmp)/std::max(cs, tmp)) > epsilon)\
  {\
    failure_count++;\
    cout << " [Failure!]" << endl;\
  }\
  else\
    cout << endl;

#define PREFIX "[C]"
  RUN_TEST("DOT", cs+=cx[i]*cy[i], 0, cs, BLAS<T>::F(clblasSdot, clblasDdot)(N, (*ds.data().handle().cl)(), 0, (*x.data().handle().cl)(), x.start()[0], x.stride()[0],
                                                                                 (*y.data().handle().cl)(), y.start()[0], y.stride()[0],
                                                                                 0, 1, &clqueue, 0, NULL, NULL));

  RUN_TEST("ASUM", cs+=std::fabs(cx[i]), 0, cs, BLAS<T>::F(clblasSasum, clblasDasum)(N, (*ds.data().handle().cl)(), 0, (*x.data().handle().cl)(), x.start()[0], x.stride()[0],
                                                                                             0, 1, &clqueue, 0, NULL, NULL));
#undef PREFIX
#define PREFIX "[C++]"

  RUN_TEST("s = x'.y", cs+=cx[i]*cy[i], 0, cs, ds = dot(x,y));
  RUN_TEST("s = exp(x'.y)", cs += cx[i]*cy[i], 0, std::exp(cs), ds = exp(dot(x,y)));
  RUN_TEST("s = 1 + x'.y", cs += cx[i]*cy[i], 0, 1 + cs, ds = 1 + dot(x,y));
  RUN_TEST("s = x'.y + y'.y", cs+= cx[i]*cy[i] + cy[i]*cy[i], 0, cs, ds = dot(x,y) + dot(y,y));
  RUN_TEST("s = max(x)", cs = std::max(cs, cx[i]), -INFINITY, cs, ds = max(x));
  RUN_TEST("s = min(x)", cs = std::min(cs, cx[i]), INFINITY, cs, ds = min(x));

#undef RUN_TEST

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename T>
void test_impl(T epsilon, ad::driver::Context const & ctx)
{
  using isaac::_;

  int_t N = 24378;
  int_t SUBN = 531;

  INIT_VECTOR(N, SUBN, 2, 4, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 5, 8, cy, y, ctx);

#define TEST_OPERATIONS(TYPE)\
  test_reduction(epsilon, cx_ ## TYPE, cy_ ## TYPE,\
                                    x_ ## TYPE, y_ ## TYPE);\

  std::cout << "> standard..." << std::endl;
  TEST_OPERATIONS(full);
  std::cout << "> slice..." << std::endl;
  TEST_OPERATIONS(slice);
}

int main()
{
  auto data = ad::driver::queues.contexts();
  for(const auto & elem : data)
  {
    ad::driver::Device device = elem.second[0].device();
    std::cout << "Device: " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
    std::cout << "---" << std::endl;
    std::cout << ">> float" << std::endl;
    test_impl<float>(1e-4, elem.first);
    std::cout << ">> double" << std::endl;
    test_impl<double>(1e-9, elem.first);
    std::cout << "---" << std::endl;
  }
  return EXIT_SUCCESS;
}
