#include <cmath>
#include <iostream>

#include "common.hpp"
#include "isaac/array.h"
#include "isaac/wrap/clBLAS.h"

namespace sc = isaac;
typedef sc::int_t int_t;

template<typename T>
void test_impl(T epsilon,  simple_vector_base<T> & cx, simple_vector_base<T> & cy,
                                sc::array & x, sc::array & y, interface_t interf)
{
  using namespace std;
  sc::driver::Context const & ctx = x.context();
  int_t N = cx.size();
  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(ctx,0);
  sc::array scratch(N, x.dtype());

  unsigned int failure_count = 0;

  isaac::numeric_type dtype = sc::to_numeric_type<T>::value;

  T cs = 0;
  T tmp = 0;
  isaac::scalar ds(dtype, ctx);

#define RUN_TEST(NAME, CPU_REDUCTION, INIT, ASSIGNMENT, GPU_REDUCTION) \
  cout <<  NAME "..." << flush;\
  cs = INIT;\
  for(int_t i = 0 ; i < N ; ++i)\
    CPU_REDUCTION;\
  cs= ASSIGNMENT ;\
  GPU_REDUCTION;\
  queue.synchronize();\
  tmp = ds;\
  if((std::abs(cs - tmp)/std::max(cs, tmp)) > epsilon)\
  {\
    failure_count++;\
    cout << " [Failure!]" << endl;\
  }\
  else\
    cout << endl;


  if(ctx.backend()==sc::driver::OPENCL && interf==clBLAS)
  {
      cl_command_queue clqueue = queue.handle().cl();

      RUN_TEST("DOT", cs+=cx[i]*cy[i], 0, cs, BLAS<T>::F(clblasSdot, clblasDdot)(N, CHANDLE(ds), 0, CHANDLE(x), x.start()[0], x.stride()[0],
                                                                                     CHANDLE(y), y.start()[0], y.stride()[0],
                                                                                     CHANDLE(scratch), 1, &clqueue, 0, NULL, NULL));
      RUN_TEST("ASUM", cs+=std::fabs(cx[i]), 0, cs, BLAS<T>::F(clblasSasum, clblasDasum)(N, CHANDLE(ds), 0, CHANDLE(x), x.start()[0], x.stride()[0],
                                                                                                CHANDLE(scratch), 1, &clqueue, 0, NULL, NULL));
  }


  RUN_TEST("s = x'.y", cs+=cx[i]*cy[i], 0, cs, ds = dot(x,y));
  RUN_TEST("s = exp(x'.y)", cs += cx[i]*cy[i], 0, std::exp(cs), ds = exp(dot(x,y)));
  RUN_TEST("s = 1 + x'.y", cs += cx[i]*cy[i], 0, 1 + cs, ds = 1 + dot(x,y));
  RUN_TEST("s = x'.y + y'.y", cs+= cx[i]*cy[i] + cy[i]*cy[i], 0, cs, ds = dot(x,y) + dot(y,y));
  RUN_TEST("s = max(x)", cs = std::max(cs, cx[i]), std::numeric_limits<T>::min(), cs, ds = max(x));
  RUN_TEST("s = min(x)", cs = std::min(cs, cx[i]), std::numeric_limits<T>::max(), cs, ds = min(x));

#undef RUN_TEST

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename T>
void test(T epsilon, sc::driver::Context const & ctx)
{
  int_t N = 10007;
  int_t SUBN = 7;

  INIT_VECTOR(N, SUBN, 0, 1, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 0, 1, cy, y, ctx);

#define TEST_OPERATIONS(TYPE, ITF)\
  test_impl(epsilon, cx_ ## TYPE, cy_ ## TYPE,\
                                    x_ ## TYPE, y_ ## TYPE, ITF);\

  std::cout << "> standard..." << std::endl;
  TEST_OPERATIONS(full, clBLAS);
  TEST_OPERATIONS(full, CPP);
  std::cout << "> slice..." << std::endl;
  TEST_OPERATIONS(slice, clBLAS);
  TEST_OPERATIONS(slice, CPP);
}


int main()
{
  clblasSetup();
  std::list<isaac::driver::Context const *> data;
  sc::driver::backend::contexts::get(data);
  for(isaac::driver::Context const * context : data)
  {
    sc::driver::Device device = sc::driver::backend::queues::get(*context,0).device();
    std::cout << "Device: " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
    std::cout << "---" << std::endl;
    std::cout << ">> float" << std::endl;
    test<float>(eps_float, *context);
    if(device.fp64_support())
    {
        std::cout << ">> double" << std::endl;
        test<double>(eps_double, *context);
    }
    std::cout << "---" << std::endl;
  }
  clblasTeardown();
  return EXIT_SUCCESS;
}
