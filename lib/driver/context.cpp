/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
