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

#ifndef ISAAC_DRIVER_BUFFER_H
#define ISAAC_DRIVER_BUFFER_H

#include "isaac/types.h"
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/handle.h"
#include "isaac/driver/dispatch.h"
namespace isaac
{

namespace driver
{

// Buffer
class ISAACAPI Buffer
{
public:
  typedef HANDLE_TYPE(cl_mem, CUdeviceptr) handle_type;

private:
  friend class CommandQueue;
  friend class Kernel;

  static CUcontext context(CUdeviceptr h)
  {
      CUcontext res;
      cuda::check(dispatch::cuPointerGetAttribute((void*)&res, CU_POINTER_ATTRIBUTE_CONTEXT, h));
      return res;
  }

public:
  Buffer(CUdeviceptr h = 0, bool take_ownership = true);
  Buffer(cl_mem Buffer = 0, bool take_ownership = true);
  Buffer(Context const & context, size_t size);
  Context const & context() const;
  bool operator<(Buffer const &) const;
  bool operator==(Buffer const &) const;
  handle_type&  handle();
  handle_type const &  handle() const;
private:
  backend_type backend_;
  Context context_;
  handle_type h_;
};

}

}

#endif
