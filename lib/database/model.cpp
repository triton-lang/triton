#include <set>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <memory>
#include <string>

#include "isaac/kernels/parse.h"
#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/templates/dot.h"
#include "isaac/kernels/templates/ger.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/kernels/templates/gemm.h"
#include "isaac/driver/program_cache.h"
#include "isaac/exception/unknown_datatype.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/database/model.h"

#include "getenv.hpp"
#include "to_string.hpp"

namespace isaac
{

static long time_event(long sum, driver::Event const & e)
{
    return sum + e.elapsed_time();
}

void model::fill_program_name(char* program_name, expressions_tuple const & expressions, binding_policy_t binding_policy)
{
  if (expressions.order()==expressions_tuple::INDEPENDENT)
    *program_name++='i';
  else
    *program_name++='s';
  symbolic_binder* binder = NULL;
  if(binding_policy==BIND_TO_HANDLE)
    binder = new bind_to_handle();
  else
    binder = new bind_all_unique();
  for (const auto & elem : expressions.data())
    traverse(*elem, elem->root(), array_expression_representation_functor(*binder, program_name),true);
  *program_name='\0';
  delete binder;
}

driver::Program const & model::init(controller<expressions_tuple> const & expressions)
{
  driver::Context & context = (driver::Context&)expressions.x().context();
  std::string pname;
  compilation_options_type const & opt = expressions.compilation_options();
  if(opt.program_name.empty())
  {
    char program_name[256];
    fill_program_name(program_name, expressions.x(), BIND_TO_HANDLE);
    pname = std::string(program_name);
  }
  else
    pname = expressions.compilation_options().program_name;

  driver::Program const * program = cache_.find(pname);
  if(program)
      return *program;

  std::string srcs;
   for(unsigned int i = 0 ; i < templates_.size() ; ++i){
     srcs += templates_[i]->generate(tools::to_string(i), expressions.x(), context.device());
   }
   srcs += fallback_->generate("fallback", expressions.x(), context.device());
   return cache_.add(context, pname, srcs);
}

model::model(expression_type etype, numeric_type dtype, predictors::random_forest const & predictor, std::vector< std::shared_ptr<templates::base> > const & templates, driver::CommandQueue const & queue) :
  templates_(templates), fallback_(fallbacks[std::make_pair(etype, dtype)]), predictor_(new predictors::random_forest(predictor)), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}


model::model(expression_type etype, numeric_type dtype, templates::base const & tp, driver::CommandQueue const & queue) : templates_(1,tp.clone()), fallback_(fallbacks[std::make_pair(etype, dtype)]), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}

void model::execute(controller<expressions_tuple> const & expr)
{
  driver::Program const & program = init(expr);
  std::vector<int_t> x = templates_[0]->input_sizes(expr.x());

  //Specific tuning if requested
  if(expr.dispatcher_options().tune && hardcoded_.find(x)==hardcoded_.end())
  {
    std::vector<double> timings(templates_.size());
    for(unsigned int i = 0 ; i < templates_.size() ; ++i)
    {
      std::list<driver::Event> events;
      try{
        templates_[i]->enqueue(queue_, program, tools::to_string(i), *fallback_, control(expr.x(), execution_options_type(0, &events)));
        queue_.synchronize();
        timings[i] = 1e-9*std::accumulate(events.begin(), events.end(), 0, &time_event);
      }catch(...){
        timings[i] = INFINITY;
      }
    }
    //Fill the override
    std::vector<int_t> x = templates_[0]->input_sizes(expr.x());
    hardcoded_[x] = std::distance(timings.begin(),std::min_element(timings.begin(), timings.end()));
  }

  //Prediction
  int label = 0;
  if(expr.dispatcher_options().label>=0)
    label = expr.dispatcher_options().label;
  else  if(hardcoded_.find(x)!=hardcoded_.end())
    label = hardcoded_.at(x);
  else if(predictor_.get())
  {
    std::vector<float> predictions = predictor_->predict(x);
    do{
        label = std::distance(predictions.begin(),std::max_element(predictions.begin(), predictions.end()));
        predictions[label] = 0;
    }while(templates_[label]->is_invalid(expr.x(),queue_.device()));
  }

  //Execution
//  std::cout << std::endl << "Label: " << label << std::endl;
  return templates_[label]->enqueue(queue_, program, tools::to_string(label), *fallback_, expr);
}

model::templates_container const & model::templates() const
{
    return templates_;
}

///////////////////

//

std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<templates::base> > init_fallback()
{
  typedef std::shared_ptr<templates::base> ptr_t;
  std::map<std::pair<expression_type, numeric_type>, ptr_t > res;
  numeric_type types[] = {CHAR_TYPE, UCHAR_TYPE, SHORT_TYPE, USHORT_TYPE, INT_TYPE, UINT_TYPE, LONG_TYPE, ULONG_TYPE, FLOAT_TYPE, DOUBLE_TYPE};
  for(auto DTYPE : types)
  {
    res[std::make_pair(AXPY_TYPE, DTYPE)] = ptr_t (new templates::axpy(1,64,128,templates::FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(DOT_TYPE, DTYPE)] = ptr_t(new templates::dot(1,64,128,templates::FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(GER_TYPE, DTYPE)] = ptr_t(new templates::ger(1,128,1,16,32,templates::FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(GEMV_N_TYPE, DTYPE)] = ptr_t(new templates::gemv_n(1, 8, 8, 4, 16, templates::FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(GEMV_T_TYPE, DTYPE)] = ptr_t(new templates::gemv_t(1, 8, 8, 64, 8, templates::FETCH_FROM_GLOBAL_STRIDED));
    res[std::make_pair(GEMM_NN_TYPE, DTYPE)] = ptr_t(new templates::gemm_nn(1, 8, 16, 8, 1, 8, 1, 8, templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_LOCAL, 8, 8, true));
    res[std::make_pair(GEMM_TN_TYPE, DTYPE)] = ptr_t(new templates::gemm_tn(1, 8, 16, 8, 1, 8, 1, 8, templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_LOCAL, 8, 8, true));
    res[std::make_pair(GEMM_NT_TYPE, DTYPE)] = ptr_t(new templates::gemm_nt(1, 8, 16, 8, 1, 8, 1, 8, templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_LOCAL, 8, 8, true));
    res[std::make_pair(GEMM_TT_TYPE, DTYPE)] = ptr_t(new templates::gemm_tt(1, 8, 16, 8, 1, 8, 1, 8, templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_LOCAL, 8, 8, true));
  }
  return res;
}


std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<templates::base> > fallbacks = init_fallback();

}
