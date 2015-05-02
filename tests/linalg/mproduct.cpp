#include <cmath>
#include "common.hpp"
#include "isaac/array.h"
#include "isaac/model/model.h"
namespace ad = isaac;

template<typename T>
void test_impl(T epsilon, simple_matrix_base<T> & cC, simple_matrix_base<T> const & cA, simple_matrix_base<T> const & cB,
                          ad::array & C, ad::array const & A, ad::array const & AT,  ad::array const & B, ad::array const & BT)
{
  int failure_count = 0;

  ad::int_t M = C.shape()[0];
  ad::int_t N = C.shape()[1];
  ad::int_t K = A.shape()[1];

  for(int i = 0 ; i < M ; ++i)
  {
    for(int j = 0 ; j < N ; ++j)
    {
      T cij = 0;
      for(int k = 0 ; k < K ; ++k)
        cij += cA(i,k)*cB(k,j);
      cC(i,j) = cij;
    }
  }

  std::vector<T> cCbuffer(M*N);
  for(int i = 0 ; i < M ; ++i)
    for(int j = 0 ; j < N ; ++j)
      cCbuffer[i + j*M] = cC(i,j);

  std::vector<T> buffer(M*N);
#define RUN_TEST(NAME, GPU_OP)\
  std::cout << NAME << "..." << std::flush;\
  GPU_OP;\
  ad::copy(C, buffer);\
  if(diff(buffer, cCbuffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;

  RUN_TEST("C = A * B", C = dot(A,B))
  RUN_TEST("C = A' * B", C = dot(trans(AT),B))
  RUN_TEST("C = A * B'", C = dot(A,trans(BT)))
  RUN_TEST("C = A' * B'", C = dot(trans(AT),trans(BT)))

  if(failure_count>0)
    exit(EXIT_FAILURE);
}

template<typename T>
void test_impl(T epsilon, ad::driver::Context const & ctx)
{

  int_t M = 412;
  int_t N = 248;
  int_t K = 376;

  int_t SUBM = 61;
  int_t SUBN = 75;
  int_t SUBK = 83;

  INIT_MATRIX(M, SUBM, 5, 2, N, SUBN, 7, 3, cC, C, ctx);
  INIT_MATRIX(M, SUBM, 8, 2, K, SUBK, 4, 3, cA, A, ctx);
  INIT_MATRIX(K, SUBK, 9, 4, N, SUBN, 6, 2, cB, B, ctx);
  std::cout << "full..." << std::endl;
  test_impl(epsilon, cC_full, cA_full, cB_full, C_full, A_full, AT_full, B_full, BT_full);
  std::cout << "slice..." << std::endl;
  test_impl(epsilon, cC_slice, cA_slice, cB_slice, C_slice, A_slice, AT_slice, B_slice, BT_slice);
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
