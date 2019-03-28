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

#include <map>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <memory>
#include "triton/driver/helpers/CL/infos.hpp"
#include "triton/driver/device.h"
#include "triton/driver/context.h"
#include "triton/codegen/target.h"

namespace triton
{

namespace driver
{

/* ------------------------ */
//          Host            //
/* ------------------------ */

std::unique_ptr<codegen::target> host_device::make_target() const {
  return std::unique_ptr<codegen::cpu_target>(new codegen::cpu_target());
}


/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

// maximum amount of shared memory per block
size_t ocl_device::max_shared_memory() const {
  return ocl::info<CL_DEVICE_LOCAL_MEM_SIZE>(*cl_);
}

size_t ocl_device::max_threads_per_block() const {
  return ocl::info<CL_DEVICE_MAX_WORK_ITEM_SIZES>(*cl_).at(0);
}

std::unique_ptr<codegen::target> ocl_device::make_target() const {
  return std::unique_ptr<codegen::amd_cl_target>(new codegen::amd_cl_target());
}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */

// architecture
cu_device::Architecture cu_device::nv_arch(std::pair<unsigned int, unsigned int> sm) const {
  switch(sm.first) {
   case 7:
     switch(sm.second){
     case 0: return Architecture::SM_7_0;
     }

  case 6:
    switch(sm.second){
    case 0: return Architecture::SM_6_0;
    case 1: return Architecture::SM_6_1;
    }

  case 5:
    switch(sm.second){
    case 0: return Architecture::SM_5_0;
    case 2: return Architecture::SM_5_2;
    default: return Architecture::UNKNOWN;
    }

  case 3:
    switch(sm.second){
    case 0: return Architecture::SM_3_0;
    case 5: return Architecture::SM_3_5;
    case 7: return Architecture::SM_3_7;
    default: return Architecture::UNKNOWN;
    }

  case 2:
    switch(sm.second){
    case 0: return Architecture::SM_2_0;
    case 1: return Architecture::SM_2_1;
    default: return Architecture::UNKNOWN;
    }

  default: return Architecture::UNKNOWN;
  }
}

// information query
template<CUdevice_attribute attr>
int cu_device::cuGetInfo() const{
  int res;
  dispatch::cuDeviceGetAttribute(&res, attr, *cu_);
  return res;
}

// convert to nvml
nvmlDevice_t cu_device::nvml_device() const{
  std::map<std::string, nvmlDevice_t> map;
  std::string key = pci_bus_id();
  if(map.find(key)==map.end()){
    nvmlDevice_t device;
    dispatch::nvmlDeviceGetHandleByPciBusId_v2(key.c_str(), &device);
    return map.insert(std::make_pair(key, device)).first->second;
  }
  return map.at(key);
}

// architecture
cu_device::Architecture cu_device::architecture() const{
  return nv_arch(compute_capability());
}

// number of address bits
size_t cu_device::address_bits() const{
  return sizeof(size_t)*8;
}

// name
std::string cu_device::name() const {
    char tmp[128];
    dispatch::cuDeviceGetName(tmp, 128, *cu_);
    return std::string(tmp);
}

// PCI bus ID
std::string cu_device::pci_bus_id() const{
  char tmp[128];
  dispatch::cuDeviceGetPCIBusId(tmp, 128, *cu_);
  return std::string(tmp);
}

// force the device to be interpreted as a particular cc
void cu_device::interpret_as(std::pair<size_t, size_t> cc){
  interpreted_as_ = std::make_shared<std::pair<size_t, size_t>>(cc);
}

// compute capability
std::pair<size_t, size_t> cu_device::compute_capability() const {
  if(interpreted_as_)
    return *interpreted_as_;
  size_t _major = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>();
  size_t _minor = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
  return std::make_pair(_major, _minor);
}

// maximum number of threads per block
size_t cu_device::max_threads_per_block() const {
  return cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK>();
}

// maximum amount of shared memory per block
size_t cu_device::max_shared_memory() const {
  return cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>();
}

// warp size
size_t cu_device::warp_size() const {
  return cuGetInfo<CU_DEVICE_ATTRIBUTE_WARP_SIZE>();
}


// maximum block dimensions
std::vector<size_t> cu_device::max_block_dim() const {
  std::vector<size_t> result(3);
  result[0] = cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X>();
  result[1] = cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y>();
  result[2] = cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z>();
  return result;
}

// current SM clock
size_t cu_device::current_sm_clock() const{
  unsigned int result;
  dispatch::nvmlDeviceGetClockInfo(nvml_device(), NVML_CLOCK_SM, &result);
  return result;
}

// max SM clock
size_t cu_device::max_sm_clock() const{
  unsigned int result;
  dispatch::nvmlDeviceGetMaxClockInfo(nvml_device(), NVML_CLOCK_SM, &result);
  return result;
}

// current memory clock
size_t cu_device::current_mem_clock() const{
  unsigned int result;
  dispatch::nvmlDeviceGetClockInfo(nvml_device(), NVML_CLOCK_MEM, &result);
  return result;
}

// max memory clock
size_t cu_device::max_mem_clock() const{
  unsigned int result;
  dispatch::nvmlDeviceGetMaxClockInfo(nvml_device(), NVML_CLOCK_MEM, &result);
  return result;
}

// max memory clock
void cu_device::set_max_clock() {
  dispatch::nvmlDeviceSetApplicationsClocks(nvml_device(), max_mem_clock(), max_sm_clock());
}

// print infos
std::string cu_device::infos() const{
  std::ostringstream oss;
  std::vector<size_t> max_wi_sizes = max_block_dim();
  oss << "Platform: CUDA" << std::endl;
  oss << "Name: " << name() << std::endl;
  oss << "Maximum total work-group size: " << max_threads_per_block() << std::endl;
  oss << "Maximum individual work-group sizes: " << max_wi_sizes[0] << ", " << max_wi_sizes[1] << ", " << max_wi_sizes[2] << std::endl;
  oss << "Local memory size: " << max_shared_memory() << std::endl;
  return oss.str();
}

// target
std::unique_ptr<codegen::target> cu_device::make_target() const {
  return std::unique_ptr<codegen::nvidia_cu_target>(new codegen::nvidia_cu_target());
}


}

}

