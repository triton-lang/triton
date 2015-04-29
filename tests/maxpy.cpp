#include "common.hpp"
#include "isaac/array.h"

namespace ad = isaac;
typedef isaac::int_t int_t;

template<typename T>
void test(T epsilon, simple_matrix_base<T> & cA, simple_matrix_base<T>& cB, simple_matrix_base<T>& cC, simple_vector_base<T>& cx, simple_vector_base<T>& cy,
          ad::array& A, ad::array& B, ad::array& C, ad::array& x, ad::array& y)
{
  using namespace std;

  int failure_count = 0;
  ad::numeric_type dtype = C.dtype();
  ad::driver::Context const & ctx = C.context();

  int_t M = cC.size1();
  int_t N = cC.size2();


  T aa = 3.12, bb=3.5;
  isaac::value_scalar a(aa), b(bb);
  isaac::scalar da(a, ctx), db(b, ctx);

  simple_vector<T> buffer(M*N);
#define CONVERT
#define RUN_TEST(NAME, CPU_LOOP, GPU_EXPR) \
  {\
  std::cout << NAME "..." << std::flush;\
  for(int_t i = 0 ; i < M ; ++i)\
    for(int_t j = 0 ; j < N ; ++j)\
        CPU_LOOP;\
  GPU_EXPR;\
  isaac::copy(C, buffer.data());\
  std::vector<T> cCbuffer(M*N);\
  for(int i = 0 ; i < M ; ++i)\
    for(int j = 0 ; j < N ; ++j)\
      cCbuffer[i + j*M] = cC(i,j);\
  CONVERT;\
  if(diff(cCbuffer, buffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;\
  }

  RUN_TEST("C = A", cC(i,j) = cA(i,j), C = A)
  RUN_TEST("C = A + B", cC(i,j) = cA(i,j) + cB(i,j), C = A + B)
  RUN_TEST("C = A - B", cC(i,j) = cA(i,j) - cB(i,j), C = A - B)
  RUN_TEST("C = A + B + C", cC(i,j) = cA(i,j) + cB(i,j) + cC(i,j), C = A + B + C)

  RUN_TEST("C = a*A", cC(i,j) = aa*cA(i,j), C = a*A)
  RUN_TEST("C = da*A", cC(i,j) = aa*cA(i,j), C = da*A)
  RUN_TEST("C = a*A + b*B", cC(i,j) = aa*cA(i,j) + bb*cB(i,j), C= a*A + b*B)
  RUN_TEST("C = da*A + b*B", cC(i,j) = aa*cA(i,j) + bb*cB(i,j), C= da*A + b*B)
  RUN_TEST("C = a*A + db*B", cC(i,j) = aa*cA(i,j) + bb*cB(i,j), C= a*A + db*B)
  RUN_TEST("C = da*A + db*B", cC(i,j) = aa*cA(i,j) + bb*cB(i,j), C= da*A + db*B)

  RUN_TEST("C = exp(A)", cC(i,j) = exp(cA(i,j)), C= exp(A))
  RUN_TEST("C = abs(A)", cC(i,j) = abs(cA(i,j)), C= abs(A))
  RUN_TEST("C = acos(A)", cC(i,j) = acos(cA(i,j)), C= acos(A))
  RUN_TEST("C = asin(A)", cC(i,j) = asin(cA(i,j)), C= asin(A))
  RUN_TEST("C = atan(A)", cC(i,j) = atan(cA(i,j)), C= atan(A))
  RUN_TEST("C = ceil(A)", cC(i,j) = ceil(cA(i,j)), C= ceil(A))
  RUN_TEST("C = cos(A)", cC(i,j) = cos(cA(i,j)), C= cos(A))
  RUN_TEST("C = cosh(A)", cC(i,j) = cosh(cA(i,j)), C= cosh(A))
  RUN_TEST("C = floor(A)", cC(i,j) = floor(cA(i,j)), C= floor(A))
  RUN_TEST("C = log(A)", cC(i,j) = log(cA(i,j)), C= log(A))
  RUN_TEST("C = log10(A)", cC(i,j) = log10(cA(i,j)), C= log10(A))
  RUN_TEST("C = sin(A)", cC(i,j) = sin(cA(i,j)), C= sin(A))
  RUN_TEST("C = sinh(A)", cC(i,j) = sinh(cA(i,j)), C= sinh(A))
  RUN_TEST("C = sqrt(A)", cC(i,j) = sqrt(cA(i,j)), C= sqrt(A))
  RUN_TEST("C = tan(A)", cC(i,j) = tan(cA(i,j)), C= tan(A))
  RUN_TEST("C = tanh(A)", cC(i,j) = tanh(cA(i,j)), C= tanh(A))

  RUN_TEST("C = A.*B", cC(i,j) = cA(i,j)*cB(i,j), C= A*B)
  RUN_TEST("C = A./B", cC(i,j) = cA(i,j)/cB(i,j), C= A/B)
  RUN_TEST("C = pow(A,B)", cC(i,j) = pow(cA(i,j), cB(i,j)), C= pow(A,B))

  RUN_TEST("C = eye(M, N)", cC(i,j) = i==j, C= eye(M, N, C.dtype(), C.context()))
  RUN_TEST("C = outer(x, y)", cC(i,j) = cx[i]*cy[j], C= outer(x,y))

#undef CONVERT
#define CONVERT for(int_t i = 0 ; i < M*N ; ++i) {cCbuffer[i] = !!cCbuffer[i] ; buffer[i] = !!buffer[i];}
  RUN_TEST("C = A==B", cC(i,j) = cA(i,j)==cB(i,j), C= cast(A==B, dtype))
  RUN_TEST("C = A>=B", cC(i,j) = cA(i,j)>=cB(i,j), C= cast(A>=B, dtype))
  RUN_TEST("C = A>B", cC(i,j) = cA(i,j)>cB(i,j), C= cast(A>B, dtype))
  RUN_TEST("C = A<=B", cC(i,j) = cA(i,j)<=cB(i,j), C= cast(A<=B, dtype))
  RUN_TEST("C = A<B", cC(i,j) = cA(i,j)<cB(i,j), C= cast(A<B, dtype))
  RUN_TEST("C = A!=B", cC(i,j) = cA(i,j)!=cB(i,j), C= cast(A!=B, dtype))

#undef RUN_TEST

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename T>
void test_impl(T epsilon, ad::driver::Context const & ctx)
{
  using isaac::_;

  int_t M = 1324;
  int_t N = 1143;
  int_t SUBM = 184;
  int_t SUBN = 145;

  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cA, A, ctx);
  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cB, B, ctx);
  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cC, C, ctx);
  INIT_VECTOR(M, SUBM, 5, 3, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 7, 2, cy, y, ctx);
#define TEST_OPERATIONS(TYPE)\
  test(epsilon, cA_ ## TYPE, cB_ ## TYPE, cC_ ## TYPE, cx_ ## TYPE, cy_ ## TYPE, A_ ## TYPE, B_ ## TYPE, C_ ## TYPE, x_ ## TYPE, y_ ## TYPE);\

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
