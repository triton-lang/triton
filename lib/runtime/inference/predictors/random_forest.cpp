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

#include "isaac/runtime/inference/predictors/random_forest.h"
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
