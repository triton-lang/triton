#include <cmath>
#include <algorithm>

#include "common.hpp"
#include "isaac/array.h"
#include "isaac/wrap/clBLAS.h"

namespace sc = isaac;

template<typename T>
void test_impl(T epsilon, simple_vector_base<T> & cy, simple_matrix_base<T> const & cA, simple_vector_base<T> & cx,
                                        sc::array_base & y, sc::array_base const & A, sc::array_base & x, interface_t interf, const char * prefix)
{
  int failure_count = 0;

  sc::int_t M = A.shape()[0];
  sc::int_t N = A.shape()[1];

  simple_vector<T> bufy(M);
  simple_vector<T> bufx(N);

  T alpha = static_cast<T>(4.2);
  T beta = static_cast<T>(5.6);

  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(y.context(),0);

  T yi = 0, xi = 0;
#define RUN_TEST(NAME, SIZE1, SIZE2, NEUTRAL, REDUCTION, ASSIGNMENT, GPU_REDUCTION, RES, BUF, CRES)\
  std::cout << "[" << prefix << "] \t" << NAME "..." << std::flush;\
  for(int i = 0 ; i < SIZE1 ; ++i)\
  {\
    yi = NEUTRAL;\
    xi = NEUTRAL;\
    for(int j = 0 ; j < SIZE2 ; ++j)\
      REDUCTION;\
    ASSIGNMENT;\
  }\
  GPU_REDUCTION;\
  queue.synchronize();\
  sc::copy(RES, BUF.data());\
  if(diff(CRES, BUF, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;


  if(y.context().backend()==sc::driver::OPENCL && interf==clBLAS)
  {
      cl_command_queue clqueue = queue.handle().cl();


      RUN_TEST("GEMV(ROW, NoTrans)", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = alpha*yi + beta*cy[i],
                     BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasRowMajor, clblasTrans, N, M, alpha, CHANDLE(A), OFF(A), LD(A),
                                CHANDLE(x), OFF(x), INC(x), beta, CHANDLE(y), OFF(y), INC(y),
                                1, &clqueue, 0, NULL, NULL), y, bufy, cy);

      RUN_TEST("GEMV(ROW, Trans)", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = alpha*xi + beta*cx[i],
                     BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasRowMajor, clblasNoTrans, N, M, alpha, CHANDLE(A), OFF(A), LD(A),
                                CHANDLE(y), OFF(y), INC(y), beta, CHANDLE(x), OFF(x), INC(x),
                                1, &clqueue, 0, NULL, NULL), x, bufx, cx);

      RUN_TEST("GEMV(COL, NoTrans)", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = alpha*yi + beta*cy[i],
                     BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasColumnMajor, clblasNoTrans, M, N, alpha, CHANDLE(A), OFF(A), LD(A),
                                CHANDLE(x), OFF(x), INC(x), beta, CHANDLE(y), OFF(y), INC(y),
                                1, &clqueue, 0, NULL, NULL), y, bufy, cy);

      RUN_TEST("GEMV(COL, Trans)", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = alpha*xi + beta*cx[i],
                     BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasColumnMajor, clblasTrans, M, N, alpha, CHANDLE(A), OFF(A), LD(A),
                                CHANDLE(y), OFF(y), INC(y), beta, CHANDLE(x), OFF(x), INC(x),
                                1, &clqueue, 0, NULL, NULL), x, bufx, cx);
  }
  else
  {
      RUN_TEST("x = dot(A.T, y)", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = xi, x = dot(trans(A),y), x, bufx, cx);
      RUN_TEST("x = sum(A, 0)", N, M, 0, xi+=cA(j,i), cx[i] = xi, x = sum(A,0), x, bufx, cx);
      RUN_TEST("x = max(A, 0)", N, M, std::numeric_limits<T>::min(), xi=std::max(xi,cA(j,i)), cx[i] = xi, x = max(A,0), x, bufx, cx);
      RUN_TEST("x = min(A, 0)", N, M, std::numeric_limits<T>::max(), xi=std::min(xi,cA(j,i)), cx[i] = xi, x = min(A,0), x, bufx, cx);

      RUN_TEST("y = dot(A, x)", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = yi, y = dot(A,x), y, bufy, cy);
      RUN_TEST("y = sum(A, 1)", M, N, 0, yi+=cA(i,j), cy[i] = yi, y = sum(A,1), y, bufy, cy);
      RUN_TEST("y = max(A, 1)", M, N, std::numeric_limits<T>::min(), yi=std::max(yi,cA(i,j)), cy[i] = yi, y = max(A,1), y, bufy, cy);
      RUN_TEST("y = min(A, 1)", M, N, std::numeric_limits<T>::max(), yi=std::min(yi,cA(i,j)), cy[i] = yi, y = min(A,1), y, bufy, cy);
  }

  if(failure_count>0)
    exit(EXIT_FAILURE);
}

template<typename T>
void test(T epsilon, sc::driver::Context const & ctx)
{
  int_t M = 173;
  int_t N = 241;
  int_t SUBM = 7;
  int_t SUBN = 11;

  INIT_VECTOR(M, SUBM, 7, 2, cy, y, ctx);
  INIT_VECTOR(N, SUBN, 5, 3, cx, x, ctx);

  {
      INIT_MATRIX(M, SUBM, 9, 1, N, SUBN, 8, 1, cA, A, ctx);
      test_impl(epsilon, cy, cA, cx, y, A, x, clBLAS, "BLAS, FULL");
      test_impl(epsilon, cy_s, cA_s, cx_s, y_s, A_s, x_s, clBLAS, "BLAS, SUB");
  }
  {
      INIT_MATRIX(M, SUBM, 9, 5, N, SUBN, 8, 4, cA, A, ctx);
      test_impl(epsilon, cy, cA, cx, y, A, x, CPP, "C++, FULL");
      test_impl(epsilon, cy_s, cA_s, cx_s, y_s, A_s, x_s, CPP, "C++, SUB");
  }
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
