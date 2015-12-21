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

namespace isaac{
namespace predictors{

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

#endif
