#include <fstream>
#include <algorithm>
#include <memory>
#include <numeric>

#include "rapidjson/document.h"
#include "rapidjson/to_array.hpp"

#include "isaac/driver/program_cache.h"
#include "isaac/profiles/profiles.h"
#include "isaac/kernels/parse.h"
#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/templates/dot.h"
#include "isaac/kernels/templates/ger.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/kernels/templates/gemm.h"
#include "isaac/exception/operation_not_supported.h"


#include "getenv.hpp"
#include "to_string.hpp"

namespace isaac
{

static long time_event(long sum, driver::Event const & e)
{
    return sum + e.elapsed_time();
}

driver::Program const & profiles::value_type::init(execution_handler const & expression)
{
  driver::Context & context = (driver::Context&)expression.x().context();
  std::string pname;
  compilation_options_type const & opt = expression.compilation_options();
  if(opt.program_name.empty())
  {
    char program_name[256];

    char* ptr = program_name;
    bind_independent binder;
    traverse(expression.x(), expression.x().root(), math_expression_representation_functor(binder, ptr),true);
    *ptr='\0';
    pname = std::string(program_name);

  }
  else
    pname = expression.compilation_options().program_name;

  driver::Program const * program = cache_.find(pname);

  if(program)
      return *program;

  std::string srcs;
   for(unsigned int i = 0 ; i < templates_.size() ; ++i){
     srcs += templates_[i]->generate(tools::to_string(i), expression.x(), context.device());
   }
   srcs += fallback_->generate("fallback", expression.x(), context.device());
   return cache_.add(context, pname, srcs);
}

profiles::value_type::value_type(expression_type etype, numeric_type dtype, predictors::random_forest const & predictor, std::vector< std::shared_ptr<templates::base> > const & templates, driver::CommandQueue const & queue) :
  templates_(templates), fallback_(fallbacks[std::make_pair(etype, dtype)]), predictor_(new predictors::random_forest(predictor)), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}


profiles::value_type::value_type(expression_type etype, numeric_type dtype, templates::base const & tp, driver::CommandQueue const & queue) : templates_(1,tp.clone()), fallback_(fallbacks[std::make_pair(etype, dtype)]), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}

void profiles::value_type::execute(execution_handler const & expr)
{
  driver::Program const & program = init(expr);
  std::vector<int_t> x = templates_[0]->input_sizes(expr.x());
  static const int MAX_TEMPORARY_WORKSPACE = 1e6;

  //Specific tuning if requested
  if(expr.dispatcher_options().tune && hardcoded_.find(x)==hardcoded_.end())
  {
    std::vector<double> timings(templates_.size());
    for(unsigned int i = 0 ; i < templates_.size() ; ++i)
    {
      if(templates_[i]->temporary_workspace(expr.x()) > MAX_TEMPORARY_WORKSPACE){
          timings[i] = INFINITY;
          continue;
      }
      std::list<driver::Event> events;
      try{
        templates_[i]->enqueue(queue_, program, tools::to_string(i), *fallback_, execution_handler(expr.x(), execution_options_type(0, &events)));
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
    }while(templates_[label]->temporary_workspace(expr.x()) > MAX_TEMPORARY_WORKSPACE);
  }

  //Execution
  if(templates_[label]->temporary_workspace(expr.x()) > MAX_TEMPORARY_WORKSPACE)
    throw operation_not_supported_exception("Running this operation would require an overly large temporary.");

  return templates_[label]->enqueue(queue_, program, tools::to_string(label), *fallback_, expr);
}

profiles::value_type::templates_container const & profiles::value_type::templates() const
{
    return templates_;
}


std::shared_ptr<templates::base> profiles::create(std::string const & template_name, std::vector<int> const & x)
{
  templates::fetching_policy_type fetch[] = {templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_GLOBAL_STRIDED, templates::FETCH_FROM_GLOBAL_CONTIGUOUS};
  if(template_name=="axpy")
    return std::shared_ptr<templates::base>(new templates::axpy(x[0], x[1], x[2], fetch[x[3]]));
  else if(template_name=="dot")
    return std::shared_ptr<templates::base>(new templates::dot(x[0], x[1], x[2], fetch[x[3]]));
  else if(template_name=="ger")
    return std::shared_ptr<templates::base>(new templates::ger(x[0], x[1], x[2], x[3], x[4], fetch[x[5]]));
  else if(template_name.find("gemv_n")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemv_n(x[0], x[1], x[2], x[3], x[4], fetch[x[5]]));
  else if(template_name.find("gemv_t")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemv_t(x[0], x[1], x[2], x[3], x[4], fetch[x[5]]));
  else if(template_name.find("gemm_nn")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_nn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else if(template_name.find("gemm_tn")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_tn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else if(template_name.find("gemm_nt")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_nt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else if(template_name.find("gemm_tt")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_tt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else
    throw std::invalid_argument("Invalid expression: " + template_name);
}

void profiles::import(std::string const & str, driver::CommandQueue const & queue)
{
  map_type & result = cache_[queue];
  //Parse the JSON document
  rapidjson::Document document;
  document.Parse<0>(str.c_str());
  //Deserialize
  std::vector<std::string> operations = {"axpy", "dot", "ger", "gemv_n", "gemv_t", "gemm_nn", "gemm_tn", "gemm_nt", "gemm_tt"};
  std::vector<std::string> dtype = {"float32", "float64"};
  for(auto & operation : operations)
  {
    const char * opcstr = operation.c_str();
    if(document.HasMember(opcstr))
    {
      expression_type etype = expression_type_from_string(operation);
      for(auto & elem : dtype)
      {
        const char * dtcstr = elem.c_str();
        if(document[opcstr].HasMember(dtcstr))
        {
          numeric_type dtype = numeric_type_from_string(elem);
          // Get profiles
          std::vector<std::shared_ptr<templates::base> > templates;
          rapidjson::Value const & profiles = document[opcstr][dtcstr]["profiles"];
          for (rapidjson::SizeType id = 0 ; id < profiles.Size() ; ++id)
            templates.push_back(create(operation, rapidjson::to_int_array<int>(profiles[id])));
          if(templates.size()>1)
          {
            // Get predictor
            predictors::random_forest predictor(document[opcstr][dtcstr]["predictor"]);
            result[std::make_pair(etype, dtype)] = std::shared_ptr<value_type>(new value_type(etype, dtype, predictor, templates, queue));
          }
          else
            result[std::make_pair(etype, dtype)] = std::shared_ptr<value_type>(new value_type(etype, dtype, *templates[0], queue));
        }
      }
    }
  }
}

profiles::map_type& profiles::init(driver::CommandQueue const & queue)
{
  map_type & result = cache_[queue];

  numeric_type dtypes[] = {CHAR_TYPE, UCHAR_TYPE, SHORT_TYPE, USHORT_TYPE, INT_TYPE, UINT_TYPE, LONG_TYPE, ULONG_TYPE, FLOAT_TYPE, DOUBLE_TYPE};
  expression_type etypes[] = {AXPY_TYPE, DOT_TYPE, GER_TYPE, GEMV_N_TYPE, GEMV_T_TYPE, GEMM_NN_TYPE, GEMM_NT_TYPE, GEMM_TN_TYPE, GEMM_TT_TYPE};

  for(numeric_type dtype: dtypes)
    for(expression_type etype: etypes)
      result[std::make_pair(etype, dtype)] = std::shared_ptr<value_type>(new value_type(etype, dtype, *fallbacks[std::make_pair(etype, dtype)], queue));

  driver::Device const & device = queue.device();
  presets_type::const_iterator it = presets_.find(std::make_tuple(device.type(), device.vendor(), device.architecture()));
  if(it==presets_.end()){
        //FIXME: Hadle this case
//      import(presets_.at(std::make_tuple(device.type(), device.vendor(), driver::Device::Architecture::UNKNOWN)), queue);
  }
  else
      import(it->second, queue);
  std::string homepath = tools::getenv("HOME");
  if(homepath.size())
  {
    std::string json_path = homepath + "/.isaac/devices/device0.json";
    std::ifstream t(json_path);
    if(!t)
        return result;
    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);
    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    import(str, queue);
  }

  return result;
}

profiles::map_type& profiles::get(driver::CommandQueue const & queue)
{
  std::map<driver::CommandQueue, map_type>::iterator it = cache_.find(queue);
  if(it == cache_.end())
    return init(queue);
  return it->second;
}

void profiles::set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<value_type> const & profile)
{
  cache_[queue][std::make_pair(operation,dtype)] = profile;
}

void profiles::release()
{
    cache_.clear();
}


std::map<driver::CommandQueue, profiles::map_type> profiles::cache_;

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
