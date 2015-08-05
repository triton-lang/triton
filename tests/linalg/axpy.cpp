#include <cmath>
#include <iostream>
#include "common.hpp"
#include "isaac/array.h"
#include "isaac/wrap/clBLAS.h"

namespace isc = isaac;
typedef isaac::int_t int_t;

template<typename T>
void test_element_wise_vector(T epsilon, simple_vector_base<T> & cx, simple_vector_base<T>& cy, simple_vector_base<T>& cz,
                                                 isc::array& x, isc::array& y, isc::array& z)
{
  using namespace std;

  int failure_count = 0;
  isc::numeric_type dtype = x.dtype();
  isc::driver::Context const & context = x.context();
  isc::driver::CommandQueue queue = isc::driver::backend::queues::get(context,0);
  cl_command_queue clqueue = queue.handle().cl();
  int_t N = cz.size();

  T aa = static_cast<T>(-4.3);
  T bb = static_cast<T>(3.5);
  isaac::value_scalar a(aa), b(bb);
  isaac::scalar da(a, context), db(b, context);

  simple_vector<T> buffer(N);
#define CONVERT
#define RUN_TEST_VECTOR_AXPY(NAME, CPU_LOOP, GPU_EXPR) \
  {\
  std::cout << PREFIX << " " << NAME "..." << std::flush;\
  for(int_t i = 0 ; i < N ; ++i)\
    CPU_LOOP;\
  GPU_EXPR;\
  queue.synchronize();\
  isaac::copy(z, buffer.data());\
  CONVERT;\
  if(diff(cz, buffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;\
  }

#define PREFIX "[C]"
  RUN_TEST_VECTOR_AXPY("AXPY", cz[i] = a*cx[i] + cz[i], BLAS<T>::F(clblasSaxpy, clblasDaxpy)(N, a, CHANDLE(x), x.start()[0], x.stride()[0],
                                                                                             CHANDLE(z), z.start()[0], z.stride()[0],
                                                                                             1, &clqueue, 0, NULL, NULL));

  RUN_TEST_VECTOR_AXPY("COPY", cz[i] = cx[i], BLAS<T>::F(clblasScopy, clblasDcopy)(N, CHANDLE(x), x.start()[0], x.stride()[0],
                                                                                 CHANDLE(z), z.start()[0], z.stride()[0],
                                                                                 1, &clqueue, 0, NULL, NULL));

  RUN_TEST_VECTOR_AXPY("SCAL", cz[i] = a*cz[i], BLAS<T>::F(clblasSscal, clblasDscal)(N, a, CHANDLE(z), z.start()[0], z.stride()[0],
                                                                                     1, &clqueue, 0, NULL, NULL));


#undef PREFIX
#define PREFIX "[C++]"
  RUN_TEST_VECTOR_AXPY("z = 0", cz[i] = 0, z = zeros(N, 1, dtype, context))
  RUN_TEST_VECTOR_AXPY("z = x", cz[i] = cx[i], z = x)
  RUN_TEST_VECTOR_AXPY("z = -x", cz[i] = -cx[i], z = -x)

  RUN_TEST_VECTOR_AXPY("z = x + y", cz[i] = cx[i] + cy[i], z = x + y)
  RUN_TEST_VECTOR_AXPY("z = x - y", cz[i] = cx[i] - cy[i], z = x - y)
  RUN_TEST_VECTOR_AXPY("z = x + y + z", cz[i] = cx[i] + cy[i] + cz[i], z = x + y + z)

  RUN_TEST_VECTOR_AXPY("z = a*x", cz[i] = aa*cx[i], z = a*x)
  RUN_TEST_VECTOR_AXPY("z = da*x", cz[i] = aa*cx[i], z = da*x)
  RUN_TEST_VECTOR_AXPY("z = a*x + b*y", cz[i] = aa*cx[i] + bb*cy[i], z= a*x + b*y)
  RUN_TEST_VECTOR_AXPY("z = da*x + b*y", cz[i] = aa*cx[i] + bb*cy[i], z= da*x + b*y)
  RUN_TEST_VECTOR_AXPY("z = a*x + db*y", cz[i] = aa*cx[i] + bb*cy[i], z= a*x + db*y)
  RUN_TEST_VECTOR_AXPY("z = da*x + db*y", cz[i] = aa*cx[i] + bb*cy[i], z= da*x + db*y)

  RUN_TEST_VECTOR_AXPY("z = exp(x)", cz[i] = exp(cx[i]), z= exp(x))
  RUN_TEST_VECTOR_AXPY("z = abs(x)", cz[i] = abs(cx[i]), z= abs(x))
  RUN_TEST_VECTOR_AXPY("z = acos(x)", cz[i] = acos(cx[i]), z= acos(x))
  RUN_TEST_VECTOR_AXPY("z = asin(x)", cz[i] = asin(cx[i]), z= asin(x))
  RUN_TEST_VECTOR_AXPY("z = atan(x)", cz[i] = atan(cx[i]), z= atan(x))
  RUN_TEST_VECTOR_AXPY("z = ceil(x)", cz[i] = ceil(cx[i]), z= ceil(x))
  RUN_TEST_VECTOR_AXPY("z = cos(x)", cz[i] = cos(cx[i]), z= cos(x))
  RUN_TEST_VECTOR_AXPY("z = cosh(x)", cz[i] = cosh(cx[i]), z= cosh(x))
  RUN_TEST_VECTOR_AXPY("z = floor(x)", cz[i] = floor(cx[i]), z= floor(x))
  RUN_TEST_VECTOR_AXPY("z = log(x)", cz[i] = log(cx[i]), z= log(x))
  RUN_TEST_VECTOR_AXPY("z = log10(x)", cz[i] = log10(cx[i]), z= log10(x))
  RUN_TEST_VECTOR_AXPY("z = sin(x)", cz[i] = sin(cx[i]), z= sin(x))
  RUN_TEST_VECTOR_AXPY("z = sinh(x)", cz[i] = sinh(cx[i]), z= sinh(x))
  RUN_TEST_VECTOR_AXPY("z = sqrt(x)", cz[i] = sqrt(cx[i]), z= sqrt(x))
  RUN_TEST_VECTOR_AXPY("z = tan(x)", cz[i] = tan(cx[i]), z= tan(x))
  RUN_TEST_VECTOR_AXPY("z = tanh(x)", cz[i] = tanh(cx[i]), z= tanh(x))

  RUN_TEST_VECTOR_AXPY("z = x.*y", cz[i] = cx[i]*cy[i], z= x*y)
  RUN_TEST_VECTOR_AXPY("z = x./y", cz[i] = cx[i]/cy[i], z= x/y)

  RUN_TEST_VECTOR_AXPY("z = pow(x,y)", cz[i] = pow(cx[i], cy[i]), z= pow(x,y))

#undef CONVERT
#define CONVERT for(int_t i = 0 ; i < N ; ++i) {cz[i] = !!cz[i] ; buffer[i] = !!buffer[i];}
  RUN_TEST_VECTOR_AXPY("z = x==y", cz[i] = cx[i]==cy[i], z= cast(x==y, dtype))
  RUN_TEST_VECTOR_AXPY("z = x>=y", cz[i] = cx[i]>=cy[i], z= cast(x>=y, dtype))
  RUN_TEST_VECTOR_AXPY("z = x>y", cz[i] = cx[i]>cy[i], z= cast(x>y, dtype))
  RUN_TEST_VECTOR_AXPY("z = x<=y", cz[i] = cx[i]<=cy[i], z= cast(x<=y, dtype))
  RUN_TEST_VECTOR_AXPY("z = x<y", cz[i] = cx[i]<cy[i], z= cast(x<y, dtype))
  RUN_TEST_VECTOR_AXPY("z = x!=y", cz[i] = cx[i]!=cy[i], z= cast(x!=y, dtype))
#undef RUN_TEST_VECTOR_AXPY


  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename T>
void test_impl(T epsilon, isc::driver::Context const & ctx)
{
  using isaac::_;

  int_t N = 24378;
  int_t SUBN = 531;


  INIT_VECTOR(N, SUBN, 5, 3, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 7, 8, cy, y, ctx);
  INIT_VECTOR(N, SUBN, 3, 2, cz, z, ctx);


#define TEST_OPERATIONS(TYPE)\
  test_element_wise_vector(epsilon, cx_ ## TYPE, cy_ ## TYPE, cz_ ## TYPE,\
                                    x_ ## TYPE, y_ ## TYPE, z_ ## TYPE);\

  std::cout << "> standard..." << std::endl;
  TEST_OPERATIONS(full);
  std::cout << "> slice..." << std::endl;
  TEST_OPERATIONS(slice);
}

int main()
{
  clblasSetup();
  std::list<isaac::driver::Context const *> data;
  isc::driver::backend::contexts::get(data);
  for(isaac::driver::Context const * context : data)
  {
    isc::driver::Device device = isc::driver::backend::queues::get(*context,0).device();
    if(device.type() != isc::driver::DEVICE_TYPE_GPU)
        continue;
    std::cout << "Device: " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
    std::cout << "---" << std::endl;
    std::cout << ">> float" << std::endl;
    test_impl<float>(eps_float, *context);
    if(device.fp64_support())
    {
        std::cout << ">> double" << std::endl;
        test_impl<double>(eps_double, *context);
    }
    std::cout << "---" << std::endl;
  }
  clblasTeardown();
  return EXIT_SUCCESS;
}
