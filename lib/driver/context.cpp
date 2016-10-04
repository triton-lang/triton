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

#include "isaac/driver/context.h"
#include "isaac/driver/program.h"

#include "helpers/ocl/infos.hpp"

#include "isaac/tools/sys/getenv.hpp"
#include "isaac/tools/sys/mkdir.hpp"

namespace isaac
{

namespace driver
{

std::string Context::cache_path()
{
    //user-specified cache path
    std::string result = tools::getenv("ISAAC_CACHE_PATH");
    if(!result.empty()){
        if(tools::mkpath(result)==0)
            return result;
    }

    //create in home
    result = tools::getenv("HOME");
    if(!result.empty())
    {
        result = result + "/.isaac/cache/";
        if(tools::mkpath(result)==0)
            return result;
    }

    //couldn't find a directory
    return "";
}



Context::Context(CUcontext const & context, bool take_ownership) : backend_(CUDA), device_(device(context), false), cache_path_(cache_path()), h_(backend_, take_ownership)
{
    h_.cu() = context;
}

Context::Context(cl_context const & context, bool take_ownership) : backend_(OPENCL), device_(ocl::info<CL_CONTEXT_DEVICES>(context)[0], false), cache_path_(cache_path()), h_(backend_, take_ownership)
{
    h_.cl() = context;
}

Context::Context(Device const & device) : backend_(device.backend_), device_(device), cache_path_(cache_path()), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      dispatch::cuCtxCreate(&h_.cu(), CU_CTX_SCHED_AUTO, device.h_.cu());
      break;
    case OPENCL:
      cl_int err;
      h_.cl() = dispatch::clCreateContext(NULL, 1, &device_.h_.cl(), NULL, NULL, &err);
      check(err);
      break;
    default:
      throw;
  }
}

Context::handle_type const & Context::handle() const
{ return h_; }

Device const & Context::device() const
{ return device_; }

backend_type Context::backend() const
{ return backend_; }

}
}
