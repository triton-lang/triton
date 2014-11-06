//#define VIENNACL_DEBUG_ALL

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/scheduler/execute.hpp"

#include "atidlas/tools/misc.hpp"
#include "atidlas/model/import.hpp"
#include "atidlas/model/model.hpp"

#include "common.hpp"

#include <iomanip>
#include <stdlib.h>


namespace ad = atidlas;
typedef atidlas::atidlas_int_t int_t;

template<class T>
void bench(std::map<std::string, ad::tools::shared_ptr<ad::model> > & models)
{
  typedef viennacl::matrix<T,viennacl::column_major> matrix_type;
  typedef viennacl::vector<T> vector_type;

  viennacl::tools::timer timer;
  float total_time = 0;
  std::vector<T> times;

#define BENCHMARK(OP, resname) \
  times.clear();\
  total_time = 0;\
  OP;\
  viennacl::backend::finish();\
  while(total_time < 1e-2){\
    timer.start(); \
    OP;\
    viennacl::backend::finish();\
    times.push_back(timer.get());\
    total_time += times.back();\
  }\
  viennacl::backend::finish();\
  float resname = ad::tools::median(times);

#define BENCH(declarations, statement_op, sizes, measure, N, key) \
    if(models.find(key)!=models.end()){\
        if(!first)\
        {\
          std::cout << std::endl;\
          std::cout << std::endl;\
        }\
        std::cout << "#"  << key << std::endl;\
        for(std::vector<int_t>::const_iterator it = sizes.begin() ; it != sizes.end() ; ++it)\
        {\
          declarations;\
          viennacl::scheduler::statement statement(statement_op);\
          BENCHMARK(models.at(key)->execute(statement), time_model);\
          BENCHMARK(models[key]->execute(statement, true), time_unique_kernel);\
          models[key]->tune(statement);\
          BENCHMARK(models[key]->execute(statement), time_opt);\
          std::cout << *it << " " << measure<T>(N,time_unique_kernel) << " " << measure<T>(N,time_model) << " " << measure<T>(N,time_opt) << std::endl;\
        }\
    }\

#define DECLARE(type, ...) type __VA_ARGS__
#define ARGS(...) __VA_ARGS__

  /*---------*/
  /*--BLAS1--*/
  /*---------*/

  //AXPY
  bool first =true;
  BENCH(DECLARE(viennacl::vector<T>, x(*it), y(*it)), ARGS(y, viennacl::op_assign(), x + y),
        BLAS1_N, bandwidth, 3*(*it), "vector-axpy-float32");
  first=false;


  //DOT
  BENCH(DECLARE(viennacl::scalar<T> s(0)); DECLARE(vector_type, x(*it), y(*it)), ARGS(s, viennacl::op_assign(), viennacl::linalg::inner_prod(x,y)),
          BLAS1_N, bandwidth, 2*(*it), "reduction-float32");


  /*---------*/
  /*--BLAS2--*/
  /*---------*/

  //N-layout
  for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
  {
      BENCH(DECLARE(matrix_type, A(*Mit,*it)); DECLARE(vector_type, y(*Mit), x(*it)),ARGS(y, viennacl::op_assign(), viennacl::linalg::prod(A,x)), BLAS2_N,
             bandwidth, (*Mit)*(*it), "row-wise-reductionN-float32");
  }


  //T-layout
  for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
  {
      BENCH(DECLARE(matrix_type, A(*it,*Mit)) ; DECLARE(vector_type, y(*Mit), x(*it)), ARGS(y, viennacl::op_assign(), viennacl::linalg::prod(viennacl::trans(A),x)), BLAS2_N,
             bandwidth, (*Mit)*(*it), "row-wise-reductionT-float32");
  }

  /*---------*/
  /*--BLAS3--*/
  /*---------*/
}

int main(int argc, char* argv[])
{
  if(argc != 2)
  {
      std::cerr << "Usage : PROG model_file" << std::endl;
      exit(EXIT_FAILURE);
  }
  std::map<std::string, ad::tools::shared_ptr<ad::model> > models = ad::import(argv[1]);

  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(models);
}
