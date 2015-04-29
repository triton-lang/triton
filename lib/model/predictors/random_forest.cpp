#include "isaac/model/predictors/random_forest.h"
#include "../convert.hpp"
namespace isaac
{

namespace predictors
{


random_forest::tree::tree(rapidjson::Value const & treerep)
{
  children_left_ = tools::to_int_array<int>(treerep["children_left"]);
  children_right_ = tools::to_int_array<int>(treerep["children_right"]);
  threshold_ = tools::to_float_array<float>(treerep["threshold"]);
  feature_ = tools::to_float_array<float>(treerep["feature"]);
  for(rapidjson::SizeType i = 0 ; i < treerep["value"].Size() ; i++)
    value_.push_back(tools::to_float_array<float>(treerep["value"][i]));
  D_ = value_[0].size();
}

std::vector<float> const & random_forest::tree::predict(std::vector<int_t> const & x) const
{
  int_t idx = 0;
  while(children_left_[idx]!=-1)
    idx = (x[feature_[idx]] <= threshold_[idx])?children_left_[idx]:children_right_[idx];
  return value_[idx];
}

int_t random_forest::tree::D() const { return D_; }

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
    for(int_t i = 0 ; i < D_ ; ++i)
      res[i] += subres[i];
  }
  for(int_t i = 0 ; i < D_ ; ++i)
    res[i] /= estimators_.size();
  return res;
}

std::vector<random_forest::tree> const & random_forest::estimators() const
{ return estimators_; }

}

}
