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

#ifndef ISAAC_DRIVER_EVENT_H
#define ISAAC_DRIVER_EVENT_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

// Event
class ISAACAPI Event
{
  friend class CommandQueue;
public:
  Event(cl_event const & event, bool take_ownership = true);
  Event(backend_type backend);
  long elapsed_time() const;
  HANDLE_TYPE(cl_event, cu_event_t)& handle();
private:
  backend_type backend_;
  HANDLE_TYPE(cl_event, cu_event_t) h_;
};

}

}

#endif
