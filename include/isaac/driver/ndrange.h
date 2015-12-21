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

#ifndef ISAAC_DRIVER_NDRANGE_H
#define ISAAC_DRIVER_NDRANGE_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"

namespace isaac
{

namespace driver
{

// NDRange
class ISAACAPI NDRange
{
public:
  NDRange(size_t size0);
  NDRange(size_t size0, size_t size1);
  NDRange(size_t size0, size_t size1, size_t size2);
  int dimension() const;
  operator const size_t*() const;
private:
  size_t sizes_[3];
  int dimension_;
};

}

}

#endif
