#include "isaac/driver/platform.h"
#include "isaac/driver/device.h"
#include "helpers/ocl/infos.hpp"

#include <string>
#include "to_string.hpp"

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
      cuda::check(dispatch::cuDeviceGetCount(&N));
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
      ocl::check(dispatch::dispatch::clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices));
      std::vector<cl_device_id> device_ids(ndevices);
      ocl::check(dispatch::dispatch::clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_ALL, ndevices, device_ids.data(), NULL));
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
