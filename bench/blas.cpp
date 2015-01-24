#include "atidlas/array.h"
#include "atidlas/tools/timer.hpp"
#include "clAmdBlas.h"
#include "common.hpp"
#include "cblas.h"

#include <iomanip>
#include <stdlib.h>
#include <cmath>


namespace ad = atidlas;
typedef atidlas::int_t int_t;

void bench(ad::numeric_type dtype)
{
  unsigned int dtsize = ad::size_of(dtype);
  float total_time = 0;
  std::vector<double> times;
  ad::tools::timer timer;

#define BENCHMARK(OP, PERF) \
  {\
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
  float tres = median(times);\
  std::cout << " " << PERF(N, tres, dtsize) << std::flush;\
  }

  /*---------*/
  /*--BLAS1--*/
  /*---------*/
  std::cout << "#AXPY" << std::endl;
  for(std::vector<int_t>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
  {
    int_t N = *it;
    std::cout << N;
    /* ATIDLAS */
    atidlas::array x(N, dtype), y(N, dtype);
    BENCHMARK(y = x + y, bandwidth);
    /* clAmdBlas */
#ifdef BENCH_CLAMDBLAS
    BENCHMARK(clAmdBlasSaxpy(N, 1, x.data()(), 0, 1, y.data()(), 0, 1, 1, &atidlas::cl::get_queue(x.context(), 0)(), 0, NULL, NULL), bandwidth)
#endif
    /* BLAS */
#ifdef BENCH_CBLAS
    std::vector<float> cx(N), cy(N);
    atidlas::copy(x, cx);
    atidlas::copy(y, cy);
    BENCHMARK(cblas_saxpy(N, 1, cx.data(), 1, cy.data(), 1), bandwidth);
#endif
    std::cout << std::endl;
  }
  std::cout << "\n\n" << std::flush;


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
#ifdef BENCH_CLAMDBLAS
  clAmdBlasSetup();
#endif

  int device_idx = 0;
  if(atidlas::cl::queues.size()>1){
    atidlas::cl::queues_t & queues = atidlas::cl::queues;
    if(argc!=2)
    {
      std::cerr << "usage : blas-bench [DEVICE_IDX]" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(atidlas::cl::queues_t::const_iterator it = queues.begin() ; it != queues.end() ; ++it){
        atidlas::cl::Device device = it->first.getInfo<CL_CONTEXT_DEVICES>()[0];
        std::cout << current++ << ": " << device.getInfo<CL_DEVICE_NAME>() << "(" << atidlas::cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>() << ")" << std::endl;
      }
      exit(EXIT_FAILURE);
    }
    else if(argc==2)
      device_idx = atoi(argv[1]);
  }

  atidlas::cl::default_context_idx = device_idx;
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench(ad::FLOAT_TYPE);

#ifdef BENCH_CLAMDBLAS
  clAmdBlasTeardown();
#endif
}
