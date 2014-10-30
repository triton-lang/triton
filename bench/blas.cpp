#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/timer.hpp"

#include "atidlas/tools/misc.hpp"
#include "atidlas/model/import.hpp"
#include "atidlas/model/model.hpp"

#include <iomanip>
#include <stdlib.h>

namespace ad = atidlas;
typedef atidlas::atidlas_int_t int_t;

template<class T>
float bandwidth(std::size_t N, float t)
{
    return N * sizeof(T) * 1e-9 / t;
}

template<class T>
void bench(std::vector<int_t> BLAS1_N, std::map<std::string, ad::tools::shared_ptr<ad::model> > & models)
{
  viennacl::tools::timer timer;
  float total_time = 0;
  std::vector<T> times;

#define BENCHMARK(OP, resname) \
  times.clear();\
  total_time = 0;\
  OP;\
  viennacl::backend::finish();\
  while(total_time < 1e-1){\
    timer.start(); \
    OP;\
    viennacl::backend::finish();\
    times.push_back(timer.get());\
    total_time += times.back();\
  }\
  viennacl::backend::finish();\
  float resname = ad::tools::median(times);

  std::cout << "#N PerfNaive PerfModel PerfOpt" << std::endl;

#define BENCH(declarations, statement_op, sizes, measure, N, key) \
    std::cout << "#"  << key << std::endl;\
    for(std::vector<int_t>::const_iterator it = sizes.begin() ; it != sizes.end() ; ++it)\
    {\
      declarations;\
      viennacl::scheduler::statement statement(statement_op);\
      BENCHMARK(y = x + y, time_viennacl);\
      BENCHMARK(models[key]->execute(statement), time_model);\
      BENCHMARK(models[key]->execute(statement, true), time_unique_kernel);\
      models[key]->tune(statement);\
      BENCHMARK(models[key]->execute(statement), time_opt);\
      std::cout << *it << " " << measure<T>(N, time_viennacl) << " " << measure<T>(N,time_unique_kernel) << " " << measure<T>(N,time_model) << " " << measure<T>(N,time_opt) << std::endl;\
    }\

#define DECLARE(type, ...) type __VA_ARGS__
#define ARGS(...) __VA_ARGS__

  BENCH(DECLARE(viennacl::vector<T>, x(*it), y(*it)), ARGS(y, viennacl::op_assign(), x + y), BLAS1_N, bandwidth, 3*(*it), "vector-axpy-float32");
  std::cout << std::endl;
  std::cout << std::endl;
}

std::vector<int_t> create_log_range(int_t min, int_t max, int_t N)
{
  std::vector<int_t> res(N);
  for(int_t i = 0 ; i < N ; ++i)
    res[i] = std::exp(std::log(min) + (float)(std::log(max) - std::log(min))*i/N);
  return res;
}

int main(int argc, char* argv[])
{
  if(argc != 2)
  {
      std::cerr << "Usage : PROG model_file" << std::endl;
      exit(EXIT_FAILURE);
  }
  std::map<std::string, ad::tools::shared_ptr<ad::model> > models = ad::import(argv[1]);

  std::vector<int_t> BLAS1_N = create_log_range(1e3, 2e7, 50);

  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(BLAS1_N, models);
}
