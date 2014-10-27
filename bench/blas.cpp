#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/timer.hpp"

#include "atidlas/model/import.hpp"
#include "atidlas/model/model.hpp"

#include <iomanip>
#include <stdlib.h>

namespace ad = atidlas;
typedef atidlas::atidlas_int_t int_t;

template<class T>
void bench(std::vector<int_t> BLAS1_N, std::map<std::string, ad::tools::shared_ptr<ad::model> > & models)
{
  viennacl::tools::timer timer;
#define BENCHMARK(OP, resname) \
  OP;\
  viennacl::backend::finish();\
  timer.start(); \
  OP;\
  viennacl::backend::finish();\
  float resname = timer.get()

  //BLAS1
  {
    for(std::vector<int_t>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
    {
      viennacl::vector<T> x(*it), y(*it), z(*it);
      viennacl::scheduler::statement statement(z, viennacl::op_assign(), x + y);

      BENCHMARK(models["vector-axpy-float32"]->execute(statement), time_model);
      BENCHMARK(models["vector-axpy-float32"]->execute(statement, true), time_unique_kernel);
      models["vector-axpy-float32"]->tune(statement);
      BENCHMARK(models["vector-axpy-float32"]->execute(statement), time_opt);

      std::cout << *it << " " << time_unique_kernel << " " << time_model << " " << time_opt << std::endl;
    }
  }


}

std::vector<int_t> create_log_range(int_t min, int_t max, int_t N)
{
  std::vector<int_t> res(N);
  for(int_t i = 0 ; i < N ; ++i)
    //res[i] = std::exp(std::log(min) + float(std::log(max) - std::log(min)*i)/N);
    res[i] = std::exp(std::log(min) + (float)(std::log(max) - std::log(min))*i/N);
  return res;
}

int main()
{
  std::map<std::string, ad::tools::shared_ptr<ad::model> > models = ad::import("geforce_gt_540m.json");

  std::vector<int_t> BLAS1_N = create_log_range(1e3, 1e7, 20);

  std::cout << "Benchmark : BLAS" << std::endl;
  std::cout << "----------------" << std::endl;
  bench<float>(BLAS1_N, models);
}
