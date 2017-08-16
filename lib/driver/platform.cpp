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
      dispatch::dispatch::clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevices);
      std::vector<cl_device_id> device_ids(ndevices);
      dispatch::dispatch::clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_GPU, ndevices, device_ids.data(), NULL);
      for(cl_device_id d : device_ids)
        devices.push_back(Device(d));
      break;
    }
    default:
      throw;
  }
}

bool Platform::platforms_check(cl_platform_id platform_id) {
  if (backend_ == OPENCL) {
    cl_uint ndevices;
    dispatch::dispatch::clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevices);
    return ndevices > 0;
  } else {
    return true;
  }
}

}

}
