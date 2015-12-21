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

#ifndef ISAAC_DRIVER_PLATFORM_H
#define ISAAC_DRIVER_PLATFORM_H

#include <vector>
#include <string>

#include "isaac/defines.h"
#include "isaac/driver/common.h"

namespace isaac
{

namespace driver
{

class Device;

class ISAACAPI Platform
{
private:
public:
  Platform(backend_type);
  Platform(cl_platform_id const &);
  std::string name() const;
  std::string version() const;
  void devices(std::vector<Device> &) const;
  cl_platform_id cl_id() const;
private:
  backend_type backend_;
  cl_platform_id cl_platform_;
};

}

}

#endif
