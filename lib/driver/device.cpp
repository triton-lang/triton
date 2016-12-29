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
  dispatch::cuDeviceGetAttribute(&res, attr, h_.cu());
  return res;
}

Device::Device(CUdevice const & device, bool take_ownership): backend_(CUDA), h_(backend_, take_ownership)
{
  h_.cu() = device;
}

Device::Device(cl_device_id const & device, bool take_ownership) : backend_(OPENCL), h_(backend_, take_ownership)
{
    h_.cl() = device;
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
        case Vendor::NVIDIA:
        {
            std::pair<unsigned int, unsigned int> sm = nv_compute_capability();
            switch(sm.first)
            {
                case 6:
                  switch(sm.second)
                  {
                    case 0: return Architecture::SM_6_0;
                    case 1: return Architecture::SM_6_1;
                  }

                case 5:
                  switch(sm.second)
                  {
                    case 0: return Architecture::SM_5_0;
                    case 2: return Architecture::SM_5_2;
                    default: return Architecture::UNKNOWN;
                  }

                case 3:
                    switch(sm.second)
                    {
                      case 0: return Architecture::SM_3_0;
                      case 5: return Architecture::SM_3_5;
                      case 7: return Architecture::SM_3_7;
                      default: return Architecture::UNKNOWN;
                    }

                case 2:
                    switch(sm.second)
                    {
                      case 0: return Architecture::SM_2_0;
                      case 1: return Architecture::SM_2_1;
                      default: return Architecture::UNKNOWN;
                    }

                default: return Architecture::UNKNOWN;
            }
        }
        case Vendor::AMD:
        {
            //No simple way to query TeraScale/GCN version. Enumerate...
            std::string device_name = name();

        #define MAP_DEVICE(device,arch)if (device_name.find(device,0)!=std::string::npos) return Architecture::arch;
            //TERASCALE 2
            MAP_DEVICE("Barts",TERASCALE_2);
            MAP_DEVICE("Cedar",TERASCALE_2);
            MAP_DEVICE("Redwood",TERASCALE_2);
            MAP_DEVICE("Juniper",TERASCALE_2);
            MAP_DEVICE("Cypress",TERASCALE_2);
            MAP_DEVICE("Hemlock",TERASCALE_2);
            MAP_DEVICE("Caicos",TERASCALE_2);
            MAP_DEVICE("Turks",TERASCALE_2);
            MAP_DEVICE("WinterPark",TERASCALE_2);
            MAP_DEVICE("BeaverCreek",TERASCALE_2);

            //TERASCALE 3
            MAP_DEVICE("Cayman",TERASCALE_3);
            MAP_DEVICE("Antilles",TERASCALE_3);
            MAP_DEVICE("Scrapper",TERASCALE_3);
            MAP_DEVICE("Devastator",TERASCALE_3);

            //GCN 1
            MAP_DEVICE("Cape",GCN_1);
            MAP_DEVICE("Pitcairn",GCN_1);
            MAP_DEVICE("Tahiti",GCN_1);
            MAP_DEVICE("New Zealand",GCN_1);
            MAP_DEVICE("Curacao",GCN_1);
            MAP_DEVICE("Malta",GCN_1);

            //GCN 2
            MAP_DEVICE("Bonaire",GCN_2);
            MAP_DEVICE("Hawaii",GCN_2);
            MAP_DEVICE("Vesuvius",GCN_2);

            //GCN 3
            MAP_DEVICE("Tonga",GCN_3);
            MAP_DEVICE("Fiji",GCN_3);

            //GCN 4
            MAP_DEVICE("Polaris",GCN_4);
        #undef MAP_DEVICE

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
      dispatch::cuDeviceGetName(tmp, 128, h_.cu());
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

Device::handle_type const & Device::handle() const
{ return h_; }

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

