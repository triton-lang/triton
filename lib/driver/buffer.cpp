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
#include "isaac/driver/buffer.h"
#include "isaac/driver/backend.h"
#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{

Buffer::Buffer(CUdeviceptr h, bool take_ownership) : backend_(CUDA), context_(backend::contexts::import(Buffer::context(h))), h_(backend_, take_ownership)
{
  h_.cu() = h;
}

Buffer::Buffer(cl_mem buffer, bool take_ownership) : backend_(OPENCL), context_(backend::contexts::import(ocl::info<CL_MEM_CONTEXT>(buffer))), h_(backend_, take_ownership)
{
  h_.cl() = buffer;
}

Buffer::Buffer(Context const & context, size_t size) : backend_(context.backend_), context_(context), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      check(dispatch::cuMemAlloc(&h_.cu(), size));
      break;
    case OPENCL:
      cl_int err;
      h_.cl() = dispatch::clCreateBuffer(context.h_.cl(), CL_MEM_READ_WRITE, size, NULL, &err);
      check(err);
      break;
    default:
      throw;
  }
}

Context const & Buffer::context() const
{ return context_; }

HANDLE_TYPE(cl_mem, CUdeviceptr) & Buffer::handle()
{ return h_; }

HANDLE_TYPE(cl_mem, CUdeviceptr) const & Buffer::handle() const
{ return h_; }

}

}
