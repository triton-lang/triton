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

#include "isaac/driver/platform.h"
#include "isaac/driver/device.h"
#include "helpers/ocl/infos.hpp"

#include <string>
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

namespace driver
{

Platform::Platform(backend_type backend): backend_(backend)
{
  if(backend==CUDA)
      dispatch::cuInit(0);
}

Platform::Platform(cl_platform_id const & platform) : backend_(OPENCL)
{
  cl_platform_ = platform;
}

std::string Platform::version() const
{
  switch(backend_)
  {
    case CUDA:
      int version;
      dispatch::cuDriverGetVersion(&version);
      return tools::to_string(version);
    case OPENCL:
      return ocl::info<CL_PLATFORM_VERSION>(cl_platform_);
    default: throw;
  }
}
std::string Platform::name() const
{
  switch(backend_)
  {
    case CUDA: return "CUDA";
    case OPENCL: return ocl::info<CL_PLATFORM_NAME>(cl_platform_);
    default: throw;
  }
}

cl_platform_id Platform::cl_id() const
{
    return cl_platform_;
}

void Platform::devices(std::vector<Device> & devices) const
{
  switch(backend_)
  {
    case CUDA:
    {
      int N;
      dispatch::cuDeviceGetCount(&N);
      for(int i = 0 ; i < N ; ++i){
        CUdevice device;
        dispatch::cuDeviceGet(&device, i);
        devices.push_back(Device(device));
      }
      break;
    }
    case OPENCL:
    {
      cl_uint ndevices;
      dispatch::dispatch::clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices);
      std::vector<cl_device_id> device_ids(ndevices);
      dispatch::dispatch::clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_ALL, ndevices, device_ids.data(), NULL);
      for(cl_device_id d : device_ids)
        devices.push_back(Device(d));
      break;
    }
    default:
      throw;
  }
}

}

}
