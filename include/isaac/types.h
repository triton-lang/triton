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

#ifndef ISAAC_TYPES_H
#define ISAAC_TYPES_H

#include <algorithm>
#include <vector>
#include <cstddef>
#include "isaac/defines.h"

namespace isaac
{

typedef long long int_t;
static const int_t start = 0;
static const int_t end = -1;
static const int_t newaxis = 1;

struct slice
{
//  slice(int_t _start) : start(_start), end(_start + 1), stride(1){}
  slice(int_t _start, int_t _end, int_t _stride = 1) : start(_start), end(_end), stride(_stride) { }

  int_t size(int_t bound) const
  {
    int_t effective_end = (end < 0)?bound - std::abs(end + 1):end;
    return (effective_end - start)/stride;
  }

  int_t start;
  int_t end;
  int_t stride;
};
static const slice all = slice{start, end, 1};

}
#endif
