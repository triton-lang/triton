#include <cmath>
#include <iostream>

#include "common.hpp"
#include "atidlas/array.h"

namespace ad = atidlas;
typedef ad::int_t int_t;

template<typename T>
void test_reduction(T epsilon,  simple_vector_base<T> & cx, simple_vector_base<T> & cy,
                                ad::array & x, ad::array & y)
{
  using namespace std;

  int_t N = cx.size();
  unsigned int failure_count = 0;

  atidlas::numeric_type dtype = ad::to_numeric_type<T>::value;

  T cs = 0;
  T tmp = 0;
  atidlas::scalar ds(dtype);

#define RUN_TEST(NAME, CPU_REDUCTION, INIT, ASSIGNMENT, GPU_REDUCTION) \
  cout << NAME "..." << flush;\
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
void test_impl(T epsilon)
{
  using atidlas::_;

  int_t N = 24378;
  int_t SUBN = 531;

  INIT_VECTOR(N, SUBN, 2, 4, cx, x);
  INIT_VECTOR(N, SUBN, 5, 8, cy, y);

#define TEST_OPERATIONS(XTYPE, YTYPE)\
  test_reduction(epsilon, cx_ ## XTYPE, cy_ ## YTYPE,\
                                    x_ ## XTYPE, y_ ## YTYPE);\

  std::cout << "> standard..." << std::endl;
  TEST_OPERATIONS(vector, vector);
  std::cout << "> slice..." << std::endl;
  TEST_OPERATIONS(slice, slice);
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
