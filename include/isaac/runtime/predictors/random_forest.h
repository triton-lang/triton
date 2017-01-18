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

#ifndef ISAAC_MODEL_PREDICTORS_RANDOM_FOREST_H
#define ISAAC_MODEL_PREDICTORS_RANDOM_FOREST_H

#include <vector>
#include "isaac/types.h"

namespace rapidjson{
class CrtAllocator;
template <typename BaseAllocator> class MemoryPoolAllocator;
template <typename Encoding, typename Allocator> class GenericValue;
template<typename CharType> struct UTF8;
typedef GenericValue<UTF8<char>, MemoryPoolAllocator<CrtAllocator> > Value;
}

namespace isaac
{
namespace runtime
{
namespace predictors
{

class random_forest
{
public:
  class tree
  {
  public:
    tree(rapidjson::Value const & treerep);
    std::vector<float> const & predict(std::vector<int_t> const & x) const;
    size_t D() const;
  private:
    std::vector<int> children_left_;
    std::vector<int> children_right_;
    std::vector<float> threshold_;
    std::vector<float> feature_;
    std::vector<std::vector<float> > value_;
    size_t D_;
  };

  random_forest(rapidjson::Value const & estimators);
  std::vector<float> predict(std::vector<int_t> const & x) const;
  std::vector<tree> const & estimators() const;
private:
  std::vector<tree> estimators_;
  size_t D_;
};

}
}
}

#endif
