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

#include <iostream>

#include "isaac/driver/backend.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/event.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/driver/buffer.h"

#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{

CommandQueue::CommandQueue(cl_command_queue const & queue, bool take_ownership) : backend_(OPENCL), context_(backend::contexts::import(ocl::info<CL_QUEUE_CONTEXT>(queue))), device_(ocl::info<CL_QUEUE_DEVICE>(queue), false), h_(backend_, take_ownership)
{
  h_.cl() = queue;
}

CommandQueue::CommandQueue(Context const & context, Device const & device, cl_command_queue_properties properties): backend_(device.backend_), context_(context), device_(device), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
    {
      dispatch::cuStreamCreate(&h_.cu(), 0);
      break;
    }

    case OPENCL:
    {
      cl_int err;
      h_.cl() = dispatch::clCreateCommandQueue(context.h_.cl(), device.h_.cl(), properties, &err);
      check(err);
      break;
    }

    default:
      throw;
  }
}

backend_type CommandQueue::backend() const
{
  return backend_;
}

Context const & CommandQueue::context() const
{
  return context_;
}

Device const & CommandQueue::device() const
{
  return device_;
}

void CommandQueue::synchronize()
{
  switch(backend_)
  {
    case CUDA: dispatch::cuStreamSynchronize(h_.cu()); break;
    case OPENCL: dispatch::clFinish(h_.cl()); break;
    default: throw;
  }
}

void CommandQueue::enqueue(Kernel const & kernel, NDRange global, driver::NDRange local, std::vector<Event> const *, Event* event)
{
  switch(backend_)
  {
    case CUDA:
      if(event)
        dispatch::cuEventRecord(event->h_.cu().first, h_.cu());

      dispatch::cuLaunchKernel(kernel.h_.cu(), global[0]/local[0], global[1]/local[1], global[2]/local[2],
                    local[0], local[1], local[2], 0, h_.cu(),(void**)&kernel.cu_params_[0], NULL);

      if(event)
        dispatch::cuEventRecord(event->h_.cu().second, h_.cu());
      break;
    case OPENCL:
      dispatch::clEnqueueNDRangeKernel(h_.cl(), kernel.h_.cl(), global.dimension(), NULL, (const size_t *)global, (const size_t *) local, 0, NULL, event?&event->h_.cl():NULL);
      break;
    default: throw;
  }
}

void CommandQueue::write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr)
{
  switch(backend_)
  {
    case CUDA:
      if(blocking)
        dispatch::cuMemcpyHtoD(buffer.h_.cu() + offset, ptr, size);
      else
        dispatch::cuMemcpyHtoDAsync(buffer.h_.cu() + offset, ptr, size, h_.cu());
      break;
    case OPENCL:
      dispatch::clEnqueueWriteBuffer(h_.cl(), buffer.h_.cl(), blocking?CL_TRUE:CL_FALSE, offset, size, ptr, 0, NULL, NULL);
      break;
    default: throw;
  }
}

void CommandQueue::read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr)
{
  switch(backend_)
  {
    case CUDA:
      if(blocking)
        dispatch::cuMemcpyDtoH(ptr, buffer.h_.cu() + offset, size);
      else
        dispatch::cuMemcpyDtoHAsync(ptr, buffer.h_.cu() + offset, size, h_.cu());
      break;
    case OPENCL:
      dispatch::clEnqueueReadBuffer(h_.cl(), buffer.h_.cl(), blocking?CL_TRUE:CL_FALSE, offset, size, ptr, 0, NULL, NULL);
      break;
    default: throw;
  }
}

CommandQueue::handle_type const & CommandQueue::handle() const
{ return h_; }

CommandQueue::handle_type & CommandQueue::handle()
{ return h_; }

}

}
