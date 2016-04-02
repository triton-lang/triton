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

#ifndef ISAAC_DRIVER_HANDLE_H
#define ISAAC_DRIVER_HANDLE_H

#include <memory>

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include <iostream>
namespace isaac
{

namespace driver
{

  struct cu_event_t{
      operator bool() const { return first && second; }
      CUevent first;
      CUevent second;
  };


#define HANDLE_TYPE(CLTYPE, CUTYPE) Handle<CLTYPE, CUTYPE>

template<class CLType, class CUType>
class ISAACAPI Handle
{
private:
  static void _delete(CUcontext x);
  static void _delete(CUdeviceptr x);
  static void _delete(CUstream x);
  static void _delete(CUdevice);
  static void _delete(CUevent x);
  static void _delete(CUfunction);
  static void _delete(CUmodule x);
  static void _delete(cu_event_t x);

  static void release(cl_context x);
  static void release(cl_mem x);
  static void release(cl_command_queue x);
  static void release(cl_device_id x);
  static void release(cl_event x);
  static void release(cl_kernel x);
  static void release(cl_program x);

public:
  Handle(backend_type backend, bool take_ownership = true);
  backend_type backend() const;
  bool operator==(Handle const & other) const;
  bool operator!=(Handle const & other) const;
  bool operator<(Handle const & other) const;
  CLType & cl();
  CLType const & cl() const;
  CUType & cu();
  CUType const & cu() const;
  ~Handle();

private:
DISABLE_MSVC_WARNING_C4251
  std::shared_ptr<CLType> cl_;
  std::shared_ptr<CUType> cu_;
RESTORE_MSVC_WARNING_C4251
private:
  backend_type backend_;
  bool has_ownership_;
};

}
}

#endif
