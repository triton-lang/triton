#include <set>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <memory>

#include "rapidjson/document.h"
#include "isaac/kernels/parse.h"
#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/templates/dot.h"
#include "isaac/kernels/templates/ger.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/kernels/templates/gemm.h"
#include "isaac/driver/program_cache.h"
#include "isaac/exception/unknown_datatype.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/model/model.h"
#include "isaac/tools/make_vector.hpp"
#include "isaac/tools/timer.hpp"
#include "convert.hpp"


namespace isaac
{

static double time_event(unsigned long sum, driver::Event const & e)
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
     char buffer[16];
     sprintf(buffer,"%d",i);
     srcs += templates_[i]->generate(buffer, expressions.x(), context.device());
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
        char buffer[16];
        sprintf(buffer,"%d",i);
        templates_[i]->enqueue(queue_, program, buffer, *fallback_, control(expr.x(), execution_options_type(0, &events)));
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
    label = std::distance(predictions.begin(),std::max_element(predictions.begin(), predictions.end()));
  }

  //Execution
  char buffer[16];
  sprintf(buffer,"%d",label);
  return templates_[label]->enqueue(queue_, program, buffer, *fallback_, expr);
}

model::templates_container const & model::templates() const
{
    return templates_;
}

///////////////////

namespace detail
{
  static expression_type get_expression_type(std::string const & name)
  {
    if(name=="axpy") return AXPY_TYPE;
    if(name=="dot") return DOT_TYPE;
    if(name=="ger") return GER_TYPE;
    if(name=="gemv_n") return GEMV_N_TYPE;
    if(name=="gemv_t") return GEMV_T_TYPE;
    if(name=="gemm_nn") return GEMM_NN_TYPE;
    if(name=="gemm_nt") return GEMM_NT_TYPE;
    if(name=="gemm_tn") return GEMM_TN_TYPE;
    if(name=="gemm_tt") return GEMM_TT_TYPE;
    throw std::invalid_argument("Invalid expression: " + name);
  }

  static numeric_type get_dtype(std::string const & name)
  {
    if(name=="float32") return FLOAT_TYPE;
    if(name=="float64") return DOUBLE_TYPE;
    throw std::invalid_argument("Invalid datatype: " + name);
  }

  static std::shared_ptr<templates::base> create(std::string const & template_name, std::vector<int> const & a)
  {
    templates::fetching_policy_type fetch[] = {templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_GLOBAL_STRIDED, templates::FETCH_FROM_GLOBAL_CONTIGUOUS};
    if(template_name=="axpy")
      return std::shared_ptr<templates::base>(new templates::axpy(a[0], a[1], a[2], fetch[a[3]]));
    else if(template_name=="dot")
      return std::shared_ptr<templates::base>(new templates::dot(a[0], a[1], a[2], fetch[a[3]]));
    else if(template_name=="ger")
      return std::shared_ptr<templates::base>(new templates::ger(a[0], a[1], a[2], a[3], a[4], fetch[a[5]]));
    else if(template_name.find("gemv_n")!=std::string::npos)
      return std::shared_ptr<templates::base>(new templates::gemv_n(a[0], a[1], a[2], a[3], a[4], fetch[a[5]]));
    else if(template_name.find("gemv_t")!=std::string::npos)
      return std::shared_ptr<templates::base>(new templates::gemv_t(a[0], a[1], a[2], a[3], a[4], fetch[a[5]]));
    else if(template_name.find("gemm_nn")!=std::string::npos)
      return std::shared_ptr<templates::base>(new templates::gemm_nn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else if(template_name.find("gemm_tn")!=std::string::npos)
      return std::shared_ptr<templates::base>(new templates::gemm_tn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else if(template_name.find("gemm_nt")!=std::string::npos)
      return std::shared_ptr<templates::base>(new templates::gemm_nt(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else if(template_name.find("gemm_tt")!=std::string::npos)
      return std::shared_ptr<templates::base>(new templates::gemm_tt(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], fetch[a[8]], fetch[a[9]], a[10], a[11]));
    else
      throw std::invalid_argument("Invalid expression: " + template_name);
  }
}

void models::import(std::string const & fname, driver::CommandQueue const & queue)
{
  namespace js = rapidjson;
  map_type & result = data_[queue];

  //Parse the JSON document
  js::Document document;
  std::ifstream t(fname.c_str());
  if(!t) return;
  std::string str;
  t.seekg(0, std::ios::end);
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);
  str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  document.Parse<0>(str.c_str());
  //Deserialize
  std::vector<std::string> operations = {"axpy", "dot", "ger", "gemv_n", "gemv_t", "gemm_nn", "gemm_tn", "gemm_nt", "gemm_tt"};
  std::vector<std::string> dtype = {"float32", "float64"};
  for(auto & operation : operations)
  {
    const char * opcstr = operation.c_str();
    if(document.HasMember(opcstr))
    {
      expression_type etype = detail::get_expression_type(operation);
      for(auto & elem : dtype)
      {
        const char * dtcstr = elem.c_str();
        if(document[opcstr].HasMember(dtcstr))
        {
          numeric_type dtype = detail::get_dtype(elem);

          // Get profiles
          std::vector<std::shared_ptr<templates::base> > templates;
          js::Value const & profiles = document[opcstr][dtcstr]["profiles"];
          for (js::SizeType id = 0 ; id < profiles.Size() ; ++id)
            templates.push_back(detail::create(operation, tools::to_int_array<int>(profiles[id])));

          if(templates.size()>1)
          {
            // Get predictor
            predictors::random_forest predictor(document[opcstr][dtcstr]["predictor"]);
            result[std::make_pair(etype, dtype)] = std::shared_ptr<model>(new model(etype, dtype, predictor, templates, queue));
          }
          else
            result[std::make_pair(etype, dtype)] = std::shared_ptr<model>(new model(etype, dtype, *templates[0], queue));
        }
      }
    }
  }
}

models::map_type& models::init(driver::CommandQueue const & queue)
{
  map_type & result = data_[queue];

  numeric_type dtypes[] = {CHAR_TYPE, UCHAR_TYPE, SHORT_TYPE, USHORT_TYPE, INT_TYPE, UINT_TYPE, LONG_TYPE, ULONG_TYPE, FLOAT_TYPE, DOUBLE_TYPE};
  expression_type etypes[] = {AXPY_TYPE, DOT_TYPE, GER_TYPE, GEMV_N_TYPE, GEMV_T_TYPE, GEMM_NN_TYPE, GEMM_NT_TYPE, GEMM_TN_TYPE, GEMM_TT_TYPE};

  for(numeric_type dtype: dtypes)
    for(expression_type etype: etypes)
      result[std::make_pair(etype, dtype)] = std::shared_ptr<model>(new model(etype, dtype, *fallbacks[std::make_pair(etype, dtype)], queue));

  if(const char * homepath = std::getenv("HOME"))
    import(std::string(homepath) + "/.isaac/devices/device0.json", queue);

  return result;
}

models::map_type& models::get(driver::CommandQueue const & queue)
{
  std::map<driver::CommandQueue, map_type>::iterator it = data_.find(queue);
  if(it == data_.end())
    return init(queue);
  return it->second;
}

void models::set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<model> const & model)
{
  data_[queue][std::make_pair(operation,dtype)] = model;
}

std::map<driver::CommandQueue, models::map_type> models::data_;

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
    res[std::make_pair(GER_TYPE, DTYPE)] = ptr_t(new templates::ger(1,8,8,8,8,templates::FETCH_FROM_GLOBAL_STRIDED));
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
