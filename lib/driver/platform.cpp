#include "isaac/driver/platform.h"
#include "isaac/driver/device.h"
#include "isaac/tools/to_string.hpp"
namespace isaac
{

namespace driver
{

#ifdef ISAAC_WITH_CUDA
Platform::Platform(backend_type backend): backend_(backend){}
#endif

Platform::Platform(cl::Platform const & platform) : backend_(OPENCL)
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
    case OPENCL: return cl_platform_.getInfo<CL_PLATFORM_VERSION>();
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

    case OPENCL: return cl_platform_.getInfo<CL_PLATFORM_NAME>();
    default: throw;
  }
}

std::vector<Platform> Platform::get()
{
  std::vector<Platform> result;
#ifdef ISAAC_WITH_CUDA
  result.push_back(Platform(CUDA));
#endif
  std::vector<cl::Platform> clresult;
  cl::Platform::get(&clresult);
  for(cl::Platform const & p : clresult)
    result.push_back(Platform(p));
  return result;
}

std::vector<Device> Platform::devices() const
{
  std::vector<Device> result;
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
    {
      int N;
      cuda::check(cuDeviceGetCount(&N));
      for(int i = 0 ; i < N ; ++i)
        result.push_back(Device(i));
      return result;
    }
#endif
    case OPENCL:
    {
      std::vector<cl::Device> clDevices;
      cl_platform_.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);
      for(cl::Device const & d: clDevices)
        result.push_back(Device(d));
      return result;
    }
    default:
      throw;
  }
}

}

}
