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

#ifndef ISAAC_DRIVER_DEVICE_H
#define ISAAC_DRIVER_DEVICE_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/platform.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

// Device
class ISAACAPI Device: public has_handle_comparators<Device>
{
private:
  friend class Context;
  friend class CommandQueue;

public:
  typedef Handle<cl_device_id, CUdevice> handle_type;

  //Supported types
  enum Type
  {
      GPU = CL_DEVICE_TYPE_GPU,
      CPU = CL_DEVICE_TYPE_CPU,
      ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
      UNKNOWN
  };
  //Supported vendors
  enum class Vendor
  {
      AMD,
      INTEL,
      NVIDIA,
      UNKNOWN
  };
  //Supported architectures
  enum class Architecture
  {
      //Intel
      HASWELL,
      BROADWELL,

      //NVidia
      SM_2_0,
      SM_2_1,
      SM_3_0,
      SM_3_5,
      SM_3_7,
      SM_5_0,
      SM_5_2,
      SM_6_0,
      SM_6_1,

      //AMD
      TERASCALE_2,
      TERASCALE_3,
      GCN_1,
      GCN_2,
      GCN_3,
      GCN_4,

      UNKNOWN
  };

private:
  //Metaprogramming elper to get cuda info from attribute
  template<CUdevice_attribute attr>
  int cuGetInfo() const;

public:
  //Constructors
  explicit Device(CUdevice const & device, bool take_ownership = true);
  explicit Device(cl_device_id const & device, bool take_ownership = true);
  //Accessors
  handle_type const & handle() const;
  Vendor vendor() const;
  Architecture architecture() const;
  backend_type backend() const;
  //Informations
  std::string infos() const;
  size_t clock_rate() const;
  unsigned int address_bits() const;
  driver::Platform platform() const;
  std::string name() const;
  std::string vendor_str() const;
  std::vector<size_t> max_work_item_sizes() const;
  Type type() const;
  std::string extensions() const;
  size_t max_work_group_size() const;
  size_t local_mem_size() const;
  size_t warp_wavefront_size() const;
  bool fp64_support() const;
  std::pair<unsigned int, unsigned int> nv_compute_capability() const;

private:
  backend_type backend_;
  handle_type h_;
};

}

}

#endif
