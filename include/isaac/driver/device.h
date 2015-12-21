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
class ISAACAPI Device
{
private:
  friend class Context;
  friend class CommandQueue;

public:
  enum Type
  {
    GPU = CL_DEVICE_TYPE_GPU,
    CPU = CL_DEVICE_TYPE_CPU,
    ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR
  };

  enum class Vendor
  {
      AMD,
      INTEL,
      NVIDIA,
      UNKNOWN
  };

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

      //AMD
      TERASCALE_2,
      TERASCALE_3,
      GCN_1_0,
      GCN_1_1,
      GCN_1_2,

      UNKNOWN
  };

private:
  template<CUdevice_attribute attr>
  int cuGetInfo() const;

public:
  explicit Device(CUdevice const & device, bool take_ownership = true);
  explicit Device(cl_device_id const & device, bool take_ownership = true);

  bool operator==(Device const &) const;
  bool operator<(Device const &) const;

  Vendor vendor() const;
  Architecture architecture() const;

  std::string infos() const;

  backend_type backend() const;
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
  HANDLE_TYPE(cl_device_id, CUdevice) h_;
};

}

}

#endif
