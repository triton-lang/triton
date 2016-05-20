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

#include <vector>
#include "isaac/random/rand.h"

namespace isaac
{
namespace random
{

template<typename T>
array rand(tuple const & shape, driver::Context const & context)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> rng(0, 1);

  std::vector<T> data(prod(shape));
  for(size_t i = 0 ; i < data.size() ; ++i)
    data[i] = rng(mt);

  return array(shape, data, context);
}

#define INSTANTIATE(T) template array rand<T>(tuple const &, driver::Context const & context);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}
}
