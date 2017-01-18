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

#include "isaac/runtime/predictors/random_forest.h"
#include "rapidjson/to_array.hpp"

namespace isaac
{
namespace runtime
{
namespace predictors
{


random_forest::tree::tree(rapidjson::Value const & treerep)
{
  children_left_ = rapidjson::to_int_array<int>(treerep["children_left"]);
  children_right_ = rapidjson::to_int_array<int>(treerep["children_right"]);
  threshold_ = rapidjson::to_float_array<float>(treerep["threshold"]);
  feature_ = rapidjson::to_float_array<float>(treerep["feature"]);
  for(rapidjson::SizeType i = 0 ; i < treerep["value"].Size() ; i++)
    value_.push_back(rapidjson::to_float_array<float>(treerep["value"][i]));
  D_ = value_[0].size();
}

std::vector<float> const & random_forest::tree::predict(std::vector<int_t> const & x) const
{
  int_t idx = 0;
  while(children_left_[idx]!=-1)
    idx = (x[feature_[idx]] <= threshold_[idx])?children_left_[idx]:children_right_[idx];
  return value_[idx];
}

size_t random_forest::tree::D() const { return D_; }

random_forest::random_forest(rapidjson::Value const & estimators)
{
  for(rapidjson::SizeType i = 0 ; i < estimators.Size() ; ++i)
    estimators_.push_back(tree(estimators[i]));
  D_ = estimators_.front().D();
}

std::vector<float> random_forest::predict(std::vector<int_t> const & x) const
{
  std::vector<float> res(D_, 0);
  for(const auto & elem : estimators_)
  {
    std::vector<float> const & subres = elem.predict(x);
    for(size_t i = 0 ; i < D_ ; ++i)
      res[i] += subres[i];
  }
  for(size_t i = 0 ; i < D_ ; ++i)
    res[i] /= estimators_.size();
  return res;
}

std::vector<random_forest::tree> const & random_forest::estimators() const
{ return estimators_; }

}
}
}
