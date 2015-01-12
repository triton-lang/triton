#include "atidlas/array.h"

#include "atidlas/tools/timer.hpp"

#include "common.hpp"

#include <iomanip>
#include <stdlib.h>
#include <cmath>


namespace ad = atidlas;
typedef atidlas::int_t int_t;

void bench(ad::numeric_type dtype)
{
  float total_time = 0;
  std::vector<double> times;
  ad::tools::timer timer;

#define BENCHMARK(OP, resname) \
  times.clear();\
  total_time = 0;\
  OP;\
  ad::cl::synchronize(ad::cl::default_context());\
  while(total_time < 1e-2){\
    timer.start(); \
    OP;\
    ad::cl::synchronize(ad::cl::default_context());\
    times.push_back(timer.get());\
    total_time += times.back();\
  }\
  float resname = median(times);

#define BENCH(DECLARATIONS, OPERATION, SIZES, MEASURE, N, key) \
  if(first==false)\
  {\
      std::cout << std::endl;\
      std::cout << std::endl;\
  }\
  std::cout << "#"  << key << std::endl;\
  for(std::vector<int_t>::const_iterator it = SIZES.begin() ; it != SIZES.end() ; ++it)\
  {\
    DECLARATIONS;\
    BENCHMARK(OPERATION, t0);\
    std::cout << *it << " " << t0 << " " << MEASURE(N,t0,ad::size_of(dtype)) << std::endl;\
  }\

#define DECLARE(type, ...) type __VA_ARGS__
#define ARGS(...) __VA_ARGS__

  /*---------*/
  /*--BLAS1--*/
  /*---------*/

  //AXPY
  bool first =true;
  BENCH(DECLARE(ad::array, x(*it, dtype), y(*it, dtype)), y = x + y, BLAS1_N, bandwidth, 3*(*it), "axpy");
  first=false;


//  //DOT
//  BENCH(DECLARE(ad::pointed_scalar s(dtype)); DECLARE(ad::vector, x(*it, dtype), y(*it, dtype)),  s = dot(x, y), BLAS1_N, bandwidth, 2*(*it), "dot");


//  /*---------*/
//  /*--BLAS2--*/
//  /*---------*/

//  //N-layout
//  for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
//  {
//      BENCH(DECLARE(atidlas::matrix, A(*Mit,*it)); DECLARE(atidlas::vector, y(*Mit), x(*it)),ARGS(y, viennacl::op_assign(), viennacl::linalg::prod(A,x)), BLAS2_N,
//             bandwidth, (*Mit)*(*it), "row-wise-reductionN-float32");
//  }


//  //T-layout
//  for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
//  {
//      BENCH(DECLARE(atidlas::matrix, A(*it,*Mit)) ; DECLARE(atidlas::vector, y(*Mit), x(*it)), ARGS(y, viennacl::op_assign(), viennacl::linalg::prod(viennacl::trans(A),x)), BLAS2_N,
//             bandwidth, (*Mit)*(*it), "row-wise-reductionT-float32");
//  }

//  /*---------*/
//  /*--BLAS3--*/
//  /*---------*/
}

int main(int argc, char* argv[])
{
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench(ad::FLOAT_TYPE);
}
