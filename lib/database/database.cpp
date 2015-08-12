#include <fstream>

#include "rapidjson/document.h"
#include "rapidjson/to_array.hpp"

#include "isaac/database/database.h"
#include "isaac/kernels/parse.h"
#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/templates/dot.h"
#include "isaac/kernels/templates/ger.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/kernels/templates/gemm.h"

#include "presets/broadwell.hpp"
#include "getenv.hpp"

namespace isaac
{


std::shared_ptr<templates::base> database::create(std::string const & template_name, std::vector<int> const & x)
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

void database::import(std::string const & str, driver::CommandQueue const & queue)
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
            result[std::make_pair(etype, dtype)] = std::shared_ptr<model>(new model(etype, dtype, predictor, templates, queue));
          }
          else
            result[std::make_pair(etype, dtype)] = std::shared_ptr<model>(new model(etype, dtype, *templates[0], queue));
        }
      }
    }
  }
}

database::map_type& database::init(driver::CommandQueue const & queue)
{
  map_type & result = cache_[queue];

  numeric_type dtypes[] = {CHAR_TYPE, UCHAR_TYPE, SHORT_TYPE, USHORT_TYPE, INT_TYPE, UINT_TYPE, LONG_TYPE, ULONG_TYPE, FLOAT_TYPE, DOUBLE_TYPE};
  expression_type etypes[] = {AXPY_TYPE, DOT_TYPE, GER_TYPE, GEMV_N_TYPE, GEMV_T_TYPE, GEMM_NN_TYPE, GEMM_NT_TYPE, GEMM_TN_TYPE, GEMM_TT_TYPE};

  for(numeric_type dtype: dtypes)
    for(expression_type etype: etypes)
      result[std::make_pair(etype, dtype)] = std::shared_ptr<model>(new model(etype, dtype, *fallbacks[std::make_pair(etype, dtype)], queue));

  driver::Device const & device = queue.device();
  presets_type::const_iterator it = presets_.find(std::make_tuple(device.vendor(), device.architecture()));
  if(it==presets_.end())
      import(presets_.at(std::make_tuple(device.vendor(), driver::Device::Architecture::UNKNOWN)), queue);
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

database::map_type& database::get(driver::CommandQueue const & queue)
{
  std::map<driver::CommandQueue, map_type>::iterator it = cache_.find(queue);
  if(it == cache_.end())
    return init(queue);
  return it->second;
}

void database::set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<model> const & model)
{
  cache_[queue][std::make_pair(operation,dtype)] = model;
}

std::map<driver::CommandQueue, database::map_type> database::cache_;

}
