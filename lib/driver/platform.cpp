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


#include "triton/driver/platform.h"
#include "triton/driver/device.h"

#include <string>

namespace triton
{
namespace driver
{


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

std::string cu_platform::version() const{
  int version;
  dispatch::cuDriverGetVersion(&version);
  return std::to_string(version);
}

void cu_platform::devices(std::vector<device *> &devices) const{
  int N;
  dispatch::cuDeviceGetCount(&N);
  for(int i = 0 ; i < N ; ++i){
    CUdevice dvc;
    dispatch::cuDeviceGet(&dvc, i);
    devices.push_back(new driver::cu_device(dvc));
  }
}

/* ------------------------ */
//        OpenCL            //
/* ------------------------ */

std::string cl_platform::version() const {
  size_t size;
  dispatch::clGetPlatformInfo(*cl_, CL_PLATFORM_VERSION, 0, nullptr, &size);
  std::string result(size, 0);
  dispatch::clGetPlatformInfo(*cl_, CL_PLATFORM_VERSION, size, (void*)&*result.begin(), nullptr);
  return result;
}

void cl_platform::devices(std::vector<device*> &devices) const{
  cl_uint num_devices;
  dispatch::clGetDeviceIDs(*cl_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  std::vector<cl_device_id> ids(num_devices);
  dispatch::clGetDeviceIDs(*cl_, CL_DEVICE_TYPE_GPU, num_devices, ids.data(), nullptr);
  for(cl_device_id id: ids)
    devices.push_back(new driver::ocl_device(id));
}

}
}
