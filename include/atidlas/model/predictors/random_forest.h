#ifndef ATIDLAS_MODEL_PREDICTORS_RANDOM_FOREST_H
#define ATIDLAS_MODEL_PREDICTORS_RANDOM_FOREST_H

#include <vector>
#include "atidlas/types.h"

namespace rapidjson{
class CrtAllocator;
template <typename BaseAllocator> class MemoryPoolAllocator;
template <typename Encoding, typename Allocator> class GenericValue;
template<typename CharType> struct UTF8;
typedef GenericValue<UTF8<char>, MemoryPoolAllocator<CrtAllocator> > Value;
}

namespace atidlas{
namespace predictors{

class random_forest
{
public:
  class tree
  {
  public:
    tree(rapidjson::Value const & treerep);
    std::vector<float> const & predict(std::vector<int_t> const & x) const;
    int_t D() const;
  private:
    std::vector<int> children_left_;
    std::vector<int> children_right_;
    std::vector<float> threshold_;
    std::vector<float> feature_;
    std::vector<std::vector<float> > value_;
    int_t D_;
  };

  random_forest(rapidjson::Value const & estimators);
  std::vector<float> predict(std::vector<int_t> const & x) const;
  std::vector<tree> const & estimators() const;
private:
  std::vector<tree> estimators_;
};

}
}

#endif
