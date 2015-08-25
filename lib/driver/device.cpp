#include <algorithm>
#include <sstream>
#include <cstring>
#include <memory>

#include "isaac/driver/device.h"
#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{


template<CUdevice_attribute attr>
int Device::cuGetInfo() const
{
  int res;
  cuda::check(dispatch::cuDeviceGetAttribute(&res, attr, h_.cu()));
  return res;
}

Device::Device(int ordinal): backend_(CUDA), h_(backend_, true)
{
  cuda::check(dispatch::cuDeviceGet(&h_.cu(), ordinal));
}

Device::Device(cl_device_id const & device, bool take_ownership) : backend_(OPENCL), h_(backend_, take_ownership)
{
    h_.cl() = device;
}


bool Device::operator==(Device const & other) const
{
    return h_==other.h_;
}

bool Device::operator<(Device const & other) const
{
    return h_<other.h_;
}



Device::Vendor Device::vendor() const
{
    std::string vname = vendor_str();
    std::transform(vname.begin(), vname.end(), vname.begin(), ::tolower);
    if(vname.find("nvidia")!=std::string::npos)
        return Vendor::NVIDIA;
    else if(vname.find("intel")!=std::string::npos)
        return Vendor::INTEL;
    else if(vname.find("amd")!=std::string::npos || vname.find("advanced micro devices")!=std::string::npos)
        return Vendor::AMD;
    else
        return Vendor::UNKNOWN;
}


Device::Architecture Device::architecture() const
{
    switch(vendor())
    {
        case Vendor::INTEL:
        {
            return Architecture::BROADWELL;
        }
        default:
        {
            return Architecture::UNKNOWN;
        }
    }
}

backend_type Device::backend() const
{ return backend_; }

unsigned int Device::address_bits() const
{
  switch(backend_)
  {
    case CUDA: return sizeof(size_t)*8;
    case OPENCL: return ocl::info<CL_DEVICE_ADDRESS_BITS>(h_.cl());
    default: throw;
  }

  return backend_;
}

driver::Platform Device::platform() const
{
  switch(backend_)
  {
    case CUDA: return Platform(CUDA);
    case OPENCL: return Platform(ocl::info<CL_DEVICE_PLATFORM>(h_.cl()));
    default: throw;
  }
}

std::string Device::name() const
{
  switch(backend_)
  {
    case CUDA:
      char tmp[128];
      cuda::check(dispatch::cuDeviceGetName(tmp, 128, h_.cu()));
      return std::string(tmp);
    case OPENCL:
      return ocl::info<CL_DEVICE_NAME>(h_.cl());
    default: throw;
  }
}

std::string Device::vendor_str() const
{
  switch(backend_)
  {
    case CUDA:
      return "NVidia";
    case OPENCL:
      return ocl::info<CL_DEVICE_VENDOR>(h_.cl());
    default: throw;
  }
}


std::vector<size_t> Device::max_work_item_sizes() const
{
  switch(backend_)
  {
    case CUDA:
    {
      std::vector<size_t> result(3);
      result[0] = cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X>();
      result[1] = cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y>();
      result[2] = cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z>();
      return result;
    }
    case OPENCL:
      return ocl::info<CL_DEVICE_MAX_WORK_ITEM_SIZES>(h_.cl());
    default:
      throw;
  }
}

Device::Type Device::type() const
{
  switch(backend_)
  {
    case CUDA: return Type::GPU;
    case OPENCL: return static_cast<Type>(ocl::info<CL_DEVICE_TYPE>(h_.cl()));
    default: throw;
  }
}

std::string Device::extensions() const
{
  switch(backend_)
  {
    case CUDA:
      return "";
    case OPENCL:
      return ocl::info<CL_DEVICE_EXTENSIONS>(h_.cl());
    default: throw;
  }
}

std::pair<unsigned int, unsigned int> Device::nv_compute_capability() const
{
  switch(backend_)
  {
      case OPENCL:
          return std::pair<unsigned int, unsigned int>(ocl::info<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>(h_.cl()), ocl::info<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>(h_.cl()));
      case CUDA:
          return std::pair<unsigned int, unsigned int>(cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(), cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>());
      default:
          throw;
  }
}

bool Device::fp64_support() const
{
  switch(backend_)
  {
    case OPENCL:
      return extensions().find("cl_khr_fp64")!=std::string::npos;
    case CUDA:
      return true;
    default:
      throw;
  }
}

std::string Device::infos() const
{
  std::ostringstream oss;
  std::vector<size_t> max_wi_sizes = max_work_item_sizes();

  oss << "Platform: " << platform().name() << std::endl;
  oss << "Vendor: " << vendor_str() << std::endl;
  oss << "Name: " << name() << std::endl;
  oss << "Maximum total work-group size: " << max_work_group_size() << std::endl;
  oss << "Maximum individual work-group sizes: " << max_wi_sizes[0] << ", " << max_wi_sizes[1] << ", " << max_wi_sizes[2] << std::endl;
  oss << "Local memory size: " << local_mem_size() << std::endl;

  return oss.str();
}

// Properties
#define WRAP_ATTRIBUTE(ret, fname, CUNAME, CLNAME) \
  ret Device::fname() const\
  {\
    switch(backend_)\
    {\
      case CUDA: return cuGetInfo<CUNAME>();\
      case OPENCL: return static_cast<ret>(ocl::info<CLNAME>(h_.cl()));\
      default: throw;\
    }\
  }\


WRAP_ATTRIBUTE(size_t, max_work_group_size, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CL_DEVICE_MAX_WORK_GROUP_SIZE)
WRAP_ATTRIBUTE(size_t, local_mem_size, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CL_DEVICE_LOCAL_MEM_SIZE)
WRAP_ATTRIBUTE(size_t, warp_wavefront_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, CL_DEVICE_WAVEFRONT_WIDTH_AMD)
WRAP_ATTRIBUTE(size_t, clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, CL_DEVICE_MAX_CLOCK_FREQUENCY)



}

}

