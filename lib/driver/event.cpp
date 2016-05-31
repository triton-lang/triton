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

#include "isaac/driver/event.h"
#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{

Event::Event(backend_type backend) : backend_(backend), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      check(dispatch::dispatch::cuEventCreate(&h_.cu().first, CU_EVENT_DEFAULT));
      check(dispatch::dispatch::cuEventCreate(&h_.cu().second, CU_EVENT_DEFAULT));
      break;
    case OPENCL:
      break;
    default:
      throw;
  }
}

Event::Event(cl_event const & event, bool take_ownership) : backend_(OPENCL), h_(backend_, take_ownership)
{
  h_.cl() = event;
}

long Event::elapsed_time() const
{
  switch(backend_)
  {
    case CUDA:
      float time;
      check(dispatch::cuEventElapsedTime(&time, h_.cu().first, h_.cu().second));
      return 1e6*time;
    case OPENCL:
      return static_cast<long>(ocl::info<CL_PROFILING_COMMAND_END>(h_.cl()) - ocl::info<CL_PROFILING_COMMAND_START>(h_.cl()));
    default:
      throw;
  }
}

Event::handle_type const & Event::handle() const
{ return h_; }

}

}
