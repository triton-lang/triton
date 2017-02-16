/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <fstream>
#include <algorithm>
#include <memory>
#include <numeric>

#include "rapidjson/document.h"
#include "rapidjson/to_array.hpp"

#include "isaac/driver/program_cache.h"
#include "isaac/runtime/profiles.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/gemm.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/sys/getenv.hpp"
#include "isaac/tools/cpp/string.hpp"
#include "isaac/tools/cpp/timer.hpp"
namespace isaac
{
namespace runtime
{

driver::Program const & profiles::value_type::init(runtime::execution_handler const & expression)
{
  driver::Context & context = (driver::Context&)expression.x().context();
  std::string pname;
  runtime::compilation_options_type const & opt = expression.compilation_options();
  if(opt.program_name.empty())
    pname = symbolic::hash(expression.x());
  else
    pname = opt.program_name;
  driver::Program const * program = cache_.find(pname);

  if(program)
      return *program;

  std::string srcs;
   for(unsigned int i = 0 ; i < templates_.size() ; ++i){
     srcs += templates_[i]->generate(tools::to_string(i), expression.x(), context.device());
   }
   return cache_.add(context, pname, srcs);
}

profiles::value_type::value_type(expression_type etype, numeric_type dtype, predictors::random_forest const & predictor, std::vector< std::shared_ptr<templates::base> > const & templates, driver::CommandQueue const & queue) :
  templates_(templates), predictor_(new predictors::random_forest(predictor)), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}


profiles::value_type::value_type(numeric_type dtype, std::shared_ptr<templates::base> const & tp, driver::CommandQueue const & queue) : templates_(1,tp), queue_(queue), cache_(driver::backend::programs::get(queue,tp->type(),dtype))
{
  cache_.clear();
}

void profiles::value_type::execute(runtime::execution_handler const & expr)
{
  static const int MAX_TEMPORARY_WORKSPACE = 1e6;
  expression_tree const & tree = expr.x();
  driver::Program const & program = init(expr);
  std::vector<int_t> x = templates_[0]->input_sizes(tree);

  //Cached
  size_t label = 0;
  auto it = labels_.find(x);
  if(it!=labels_.end())
    label = it->second;
  //Not cached
  else if(predictor_)
  {
    std::vector<float> perf = predictor_->predict(x);
    std::vector<size_t> idx(perf.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&perf](size_t i1, size_t i2) {return perf[i1] > perf[i2];});
    size_t i = 0;
    while(templates_[idx[i]]->temporary_workspace(tree) > MAX_TEMPORARY_WORKSPACE)
      i++;
    label = idx[i];
    labels_.insert({x, label});
  }
  if(templates_[label]->temporary_workspace(tree) > MAX_TEMPORARY_WORKSPACE)
    throw operation_not_supported_exception("Running this operation would require an overly large temporary.");
  templates_[label]->enqueue(queue_, program, tools::to_string(label), expr);
}

profiles::value_type::templates_container const & profiles::value_type::templates() const
{
    return templates_;
}

std::shared_ptr<templates::base> profiles::create(std::string const & op, std::string const & str)
{
    if(str=="cublas_gemm"){
        if(op=="gemm_nn") return std::shared_ptr<templates::base>(new templates::cublas_gemm('N', 'N'));
        if(op=="gemm_nt") return std::shared_ptr<templates::base>(new templates::cublas_gemm('N', 'T'));
        if(op=="gemm_tn") return std::shared_ptr<templates::base>(new templates::cublas_gemm('T', 'N'));
        if(op=="gemm_tt") return std::shared_ptr<templates::base>(new templates::cublas_gemm('T', 'T'));
    }
    if(str=="intelblas_gemm"){
        if(op=="gemm_nn") return std::shared_ptr<templates::base>(new templates::intelblas_gemm('N', 'N'));
        if(op=="gemm_nt") return std::shared_ptr<templates::base>(new templates::intelblas_gemm('N', 'T'));
        if(op=="gemm_tn") return std::shared_ptr<templates::base>(new templates::intelblas_gemm('T', 'N'));
        if(op=="gemm_tt") return std::shared_ptr<templates::base>(new templates::intelblas_gemm('T', 'T'));
    }
    if(str=="intelblas_gemm_image"){
        if(op=="gemm_nn") return std::shared_ptr<templates::base>(new templates::intelblas_gemm_image('N', 'N'));
        if(op=="gemm_nt") return std::shared_ptr<templates::base>(new templates::intelblas_gemm_image('N', 'T'));
        if(op=="gemm_tn") return std::shared_ptr<templates::base>(new templates::intelblas_gemm_image('T', 'N'));
        if(op=="gemm_tt") return std::shared_ptr<templates::base>(new templates::intelblas_gemm_image('T', 'T'));
    }
    throw;
}

std::shared_ptr<templates::base> profiles::create(std::string const & template_name, std::vector<int> const & x)
{
  if(template_name=="elementwise_1d")
    return std::shared_ptr<templates::base>(new templates::elementwise_1d(x[0], x[1], x[2]));
  else if(template_name=="reduce_1d")
    return std::shared_ptr<templates::base>(new templates::reduce_1d(x[0], x[1], x[2]));
  else if(template_name=="elementwise_2d")
    return std::shared_ptr<templates::base>(new templates::elementwise_2d(x[0], x[1], x[2], x[3], x[4]));
  else if(template_name=="reduce_2d_rows")
    return std::shared_ptr<templates::base>(new templates::reduce_2d_rows(x[0], x[1], x[2], x[3], x[4]));
  else if(template_name=="reduce_2d_cols")
    return std::shared_ptr<templates::base>(new templates::reduce_2d_cols(x[0], x[1], x[2], x[3], x[4]));
  else if(template_name=="gemm_nn")
    return std::shared_ptr<templates::base>(new templates::gemm_nn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else if(template_name=="gemm_tn")
    return std::shared_ptr<templates::base>(new templates::gemm_tn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else if(template_name=="gemm_nt")
    return std::shared_ptr<templates::base>(new templates::gemm_nt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else if(template_name=="gemm_tt")
    return std::shared_ptr<templates::base>(new templates::gemm_tt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
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
  std::vector<std::string> operations = {"elementwise_1d", "reduce_1d", "elementwise_2d", "reduce_2d_rows", "reduce_2d_cols", "gemm_nn", "gemm_tn", "gemm_nt", "gemm_tt"};
  std::vector<std::string> dtype = {"float16", "float32", "float64"};
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
          for (rapidjson::SizeType i = 0 ; i < profiles.Size() ; ++i){
            if(profiles[i].IsString())
                 templates.push_back(create(operation, profiles[i].GetString()));
            else
                templates.push_back(create(operation, rapidjson::to_int_array<int>(profiles[i])));
          }
          if(templates.size()>1){
            // Get predictor
            predictors::random_forest predictor(document[opcstr][dtcstr]["predictor"]);
            result[{etype, dtype}] = std::make_shared<value_type>(etype, dtype, predictor, templates, queue);
          }
          else
            result[{etype, dtype}] = std::make_shared<value_type>(dtype, templates[0], queue);
        }
      }
    }
  }
}

profiles::map_type& profiles::init(driver::CommandQueue const & queue)
{
  map_type & map = cache_[queue];
  driver::Device const & device = queue.device();
  //Default
  import(presets_.at(std::make_tuple(driver::Device::Type::UNKNOWN, driver::Device::Vendor::UNKNOWN, driver::Device::Architecture::UNKNOWN)), queue);
  //Database profile
  presets_type::const_iterator it = presets_.find(std::make_tuple(device.type(), device.vendor(), device.architecture()));
  if(it!=presets_.end())
    import(it->second, queue);
  return map;
}

profiles::map_type& profiles::get(driver::CommandQueue const & queue)
{
  std::map<driver::CommandQueue, map_type>::iterator it = cache_.find(queue);
  if(it == cache_.end())
    return init(queue);
  return it->second;
}

void profiles::set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<value_type> const & profile)
{ cache_[queue][std::make_pair(operation,dtype)] = profile; }

void profiles::release()
{ cache_.clear(); }

std::map<driver::CommandQueue, profiles::map_type> profiles::cache_;

}
}
