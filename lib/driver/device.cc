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
#include "triton/driver/device.h"
#include "triton/driver/context.h"
#include "triton/driver/error.h"
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
//         CUDA             //
/* ------------------------ */

// information query
template<CUdevice_attribute attr>
int cu_device::cuGetInfo() const{
  int res;
  dispatch::cuDeviceGetAttribute(&res, attr, *cu_);
  return res;
}

// force the device to be interpreted as a particular cc
void cu_device::interpret_as(int cc){
  interpreted_as_ = std::make_shared<int>(cc);
}

// compute capability
int cu_device::compute_capability() const {
  if(interpreted_as_)
    return *interpreted_as_;
  size_t major = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>();
  size_t minor = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
  return major*10 + minor;
}

// maximum amount of shared memory per block
size_t cu_device::max_shared_memory() const {
  return cuGetInfo<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN>();
}

void cu_device::enable_peer_access(CUdeviceptr peer_mem_ptr) const{
  CUcontext context;
  dispatch::cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, peer_mem_ptr);
  try {
    dispatch::cuCtxEnablePeerAccess(context, 0);
  } catch (exception::cuda::peer_access_already_enabled) {}
}

// target
std::unique_ptr<codegen::target> cu_device::make_target() const {
  return std::unique_ptr<codegen::nvidia_cu_target>(new codegen::nvidia_cu_target(compute_capability()));
}


}

}

