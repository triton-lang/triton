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

#ifndef ISAAC_DRIVER_COMMON_H
#define ISAAC_DRIVER_COMMON_H
#include <exception>

#include "isaac/driver/dispatch.h"
#include "isaac/defines.h"


namespace isaac
{
namespace driver
{

enum backend_type
{
  OPENCL,
  CUDA
};

void check(nvrtcResult err);

void check(CUresult);
void check_destruction(CUresult);

void check(cl_int err);

}
}

#endif
