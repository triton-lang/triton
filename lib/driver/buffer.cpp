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
      dispatch::cuMemAlloc(&h_.cu(), size);
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

Buffer::handle_type & Buffer::handle()
{ return h_; }

Buffer::handle_type const & Buffer::handle() const
{ return h_; }

}

}
