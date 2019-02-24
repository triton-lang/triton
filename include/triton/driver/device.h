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

#ifndef TDL_INCLUDE_DRIVER_DEVICE_H
#define TDL_INCLUDE_DRIVER_DEVICE_H

#include "triton/driver/platform.h"
#include "triton/driver/handle.h"

namespace triton
{

namespace driver
{

// Device
class Device: public HandleInterface<Device, CUdevice>
{
public:
  //Supported architectures
  enum class Architecture{
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
    SM_7_0,
    UNKNOWN
  };

private:
  //Metaprogramming elper to get cuda info from attribute
  template<CUdevice_attribute attr>
  int cuGetInfo() const;

  inline Architecture nv_arch(std::pair<unsigned int, unsigned int> sm) const;
  inline nvmlDevice_t nvml_device() const;

public:
  Device(CUdevice cu = CUdevice(), bool take_ownership = true): cu_(cu, take_ownership){}
  //Accessors
  Architecture architecture() const;
  Handle<CUdevice> const & cu() const;
  //Informations
  std::string infos() const;
  size_t address_bits() const;
  driver::Platform platform() const;
  std::vector<size_t> max_block_dim() const;
  size_t max_threads_per_block() const;
  size_t max_shared_memory() const;
  size_t warp_size() const;
  //Compute Capability
  void interpret_as(std::pair<size_t, size_t> cc);
  std::pair<size_t, size_t> compute_capability() const;
  //Identifier
  std::string name() const;
  std::string pci_bus_id() const;
  //Clocks
  size_t current_sm_clock() const;
  size_t current_mem_clock() const;

  size_t max_sm_clock() const;
  size_t max_mem_clock() const;

private:
  Handle<CUdevice> cu_;
  std::shared_ptr<std::pair<size_t, size_t>> interpreted_as_;
};

}

}

#endif
