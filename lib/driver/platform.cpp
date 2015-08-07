#include "isaac/driver/platform.h"
#include "isaac/driver/device.h"

#include "helpers/ocl/infos.hpp"

#include <string>

namespace isaac
{

namespace driver
{

#ifdef ISAAC_WITH_CUDA
Platform::Platform(backend_type backend): backend_(backend, take_ownership){}
#endif

Platform::Platform(cl_platform_id const & platform) : backend_(OPENCL)
{
  cl_platform_ = platform;
}

std::string Platform::version() const
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      int version;
      cuDriverGetVersion(&version);
      return tools::to_string(version);
#endif
    case OPENCL: return ocl::info<CL_PLATFORM_VERSION>(cl_platform_);
    default: throw;
  }
}
std::string Platform::name() const
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA: return "CUDA";
#endif

    case OPENCL: return ocl::info<CL_PLATFORM_NAME>(cl_platform_);
    default: throw;
  }
}

void Platform::devices(std::vector<Device> & devices) const
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
    {
      int N;
      cuda::check(cuDeviceGetCount(&N));
      for(int i = 0 ; i < N ; ++i)
        devices.push_back(Device(i));
      break;
    }
#endif
    case OPENCL:
    {
      cl_uint ndevices;
      ocl::check(clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices));
      std::vector<cl_device_id> device_ids(ndevices);
      ocl::check(clGetDeviceIDs(cl_platform_, CL_DEVICE_TYPE_ALL, ndevices, device_ids.data(), NULL));
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
