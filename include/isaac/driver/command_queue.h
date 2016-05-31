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

#ifndef ISAAC_DRIVER_COMMAND_QUEUE_H
#define ISAAC_DRIVER_COMMAND_QUEUE_H

#include <map>
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class Kernel;
class Event;
class NDRange;
class Buffer;

// Command Queue
class ISAACAPI CommandQueue: public has_handle_comparators<CommandQueue>
{
public:
  typedef HANDLE_TYPE(cl_command_queue, CUstream) handle_type;

public:
  //Constructors
  CommandQueue(cl_command_queue const & queue, bool take_ownership = true);
  CommandQueue(Context const & context, Device const & device, cl_command_queue_properties properties = 0);
  //Accessors
  handle_type & handle();
  handle_type const & handle() const;
  backend_type backend() const;
  Context const & context() const;
  Device const & device() const;
  //Synchronize
  void synchronize();
  //Profiling
  void enable_profiling();
  void disable_profiling();
  //Enqueue calls
  void enqueue(Kernel const & kernel, NDRange global, driver::NDRange local, std::vector<Event> const *, Event *event);
  void write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr);

private:
  backend_type backend_;
  Context context_;
  Device device_;
  handle_type h_;
};


}

}

#endif
