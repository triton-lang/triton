/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include <fstream>
#include <algorithm>
#include <memory>
#include <numeric>

#include "rapidjson/document.h"
#include "rapidjson/to_array.hpp"

#include "isaac/driver/program_cache.h"
#include "isaac/runtime/inference/profiles.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/matrix_product.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/sys/getenv.hpp"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace runtime
{

static long time_event(long sum, driver::Event const & e)
{
    return sum + e.elapsed_time();
}

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
   for(unsigned int i = 0 ; i < templates_.size() ; ++i)
     srcs += templates_[i]->generate(tools::to_string(i), expression.x(), context.device());
   return cache_.add(context, pname, srcs);
}

profiles::value_type::value_type(expression_type etype, numeric_type dtype, predictors::random_forest const & predictor, std::vector< std::shared_ptr<templates::base> > const & templates, driver::CommandQueue const & queue) :
  templates_(templates), predictor_(new predictors::random_forest(predictor)), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}


profiles::value_type::value_type(expression_type etype, numeric_type dtype, templates::base const & tp, driver::CommandQueue const & queue) : templates_(1,tp.clone()), queue_(queue), cache_(driver::backend::programs::get(queue,etype,dtype))
{
  cache_.clear();
}

void profiles::value_type::execute(runtime::execution_handler const & expr)
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
        templates_[i]->enqueue(queue_, program, tools::to_string(i), runtime::execution_handler(expr.x(), runtime::execution_options_type(0, &events)));
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

  return templates_[label]->enqueue(queue_, program, tools::to_string(label), expr);
}

profiles::value_type::templates_container const & profiles::value_type::templates() const
{
    return templates_;
}


std::shared_ptr<templates::base> profiles::create(std::string const & template_name, std::vector<int> const & x)
{
  templates::fetch_type fetch[] = {templates::FETCH_FROM_LOCAL, templates::FETCH_FROM_GLOBAL_STRIDED, templates::FETCH_FROM_GLOBAL_CONTIGUOUS};
  if(template_name=="elementwise_1d")
    return std::shared_ptr<templates::base>(new templates::elementwise_1d(x[0], x[1], x[2], fetch[x[3]]));
  else if(template_name=="reduce_1d")
    return std::shared_ptr<templates::base>(new templates::reduce_1d(x[0], x[1], x[2], fetch[x[3]]));
  else if(template_name=="elementwise_2d")
    return std::shared_ptr<templates::base>(new templates::elementwise_2d(x[0], x[1], x[2], x[3], x[4], fetch[x[5]]));
  else if(template_name.find("reduce_2d_rows")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::reduce_2d_rows(x[0], x[1], x[2], x[3], x[4], fetch[x[5]]));
  else if(template_name.find("reduce_2d_cols")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::reduce_2d_cols(x[0], x[1], x[2], x[3], x[4], fetch[x[5]]));
  else if(template_name.find("matrix_product_nn")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::matrix_product_nn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else if(template_name.find("matrix_product_tn")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::matrix_product_tn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else if(template_name.find("matrix_product_nt")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::matrix_product_nt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
  else if(template_name.find("matrix_product_tt")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::matrix_product_tt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], fetch[x[8]], fetch[x[9]], x[10], x[11]));
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
  std::vector<std::string> operations = {"elementwise_1d", "reduce_1d", "elementwise_2d", "reduce_2d_rows", "reduce_2d_cols", "matrix_product_nn", "matrix_product_tn", "matrix_product_nt", "matrix_product_tt"};
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
  map_type & map = cache_[queue];
  driver::Device const & device = queue.device();
  presets_type::const_iterator it = presets_.find(std::make_tuple(device.type(), device.vendor(), device.architecture()));
  /*-- Device not found in database --*/
  if(it==presets_.end()){
      import(presets_.at(std::make_tuple(driver::Device::Type::UNKNOWN, driver::Device::Vendor::UNKNOWN, driver::Device::Architecture::UNKNOWN)), queue);
  }
  /*-- Device found in database --*/
  else{
      import(it->second, queue);
  }

  /*-- User-provided profile --*/
  std::string homepath = tools::getenv("HOME");
  if(homepath.size())
  {
    std::string json_path = homepath + "/.isaac/devices/device0.json";
    std::ifstream t(json_path);
    if(!t)
        return map;
    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);
    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    import(str, queue);
  }

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
