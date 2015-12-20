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
#include <algorithm>
#include "isaac/driver/ndrange.h"

namespace isaac
{

namespace driver
{

NDRange::NDRange(size_t size0)
{
    sizes_[0] = size0;
    sizes_[1] = 1;
    sizes_[2] = 1;
    dimension_ = 1;
}

NDRange::NDRange(size_t size0, size_t size1)
{
    sizes_[0] = size0;
    sizes_[1] = size1;
    sizes_[2] = 1;
    dimension_ = 2;
}

NDRange::NDRange(size_t size0, size_t size1, size_t size2)
{
    sizes_[0] = size0;
    sizes_[1] = size1;
    sizes_[2] = size2;
    dimension_ = 3;
}

int NDRange::dimension() const
{
 return dimension_;
}

NDRange::operator const size_t*() const
{
  return (const size_t*) sizes_;
}

}

}
