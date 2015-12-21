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

#ifndef ISAAC_DRIVER_CONTEXT_H
#define ISAAC_DRIVER_CONTEXT_H

#include <map>
#include <memory>
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class ISAACAPI Context
{
  friend class Program;
  friend class CommandQueue;
  friend class Buffer;
  static std::string cache_path();

  static CUdevice device(CUcontext)
  {
      CUdevice res;
      cuda::check(dispatch::cuCtxGetDevice(&res));
      return res;
  }

public:
  explicit Context(CUcontext const & context, bool take_ownership = true);
  explicit Context(cl_context const & context, bool take_ownership = true);
  explicit Context(Device const & device);

  backend_type backend() const;
  Device const & device() const;
  bool operator==(Context const &) const;
  bool operator<(Context const &) const;

  HANDLE_TYPE(cl_context, CUcontext) const & handle() const { return h_; }
private:
DISABLE_MSVC_WARNING_C4251
  backend_type backend_;
  Device device_;
  std::string cache_path_;
  HANDLE_TYPE(cl_context, CUcontext) h_;
RESTORE_MSVC_WARNING_C4251
};

}
}

#endif
