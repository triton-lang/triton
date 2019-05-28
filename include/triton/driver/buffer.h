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

#ifndef TDL_INCLUDE_DRIVER_BUFFER_H
#define TDL_INCLUDE_DRIVER_BUFFER_H

#include "triton/driver/handle.h"
#include "triton/driver/context.h"

namespace triton
{
namespace driver
{

class stream;

// Base
class buffer : public polymorphic_resource<CUdeviceptr, cl_mem, host_buffer_t> {
public:
  buffer(driver::context* ctx, CUdeviceptr cl, bool take_ownership);
  buffer(driver::context* ctx, cl_mem cl, bool take_ownership);
  buffer(driver::context* ctx, host_buffer_t hst, bool take_ownership);
  static buffer* create(driver::context* ctx, size_t size);
  driver::context* context();

protected:
  driver::context* context_;
};

// CPU
class host_buffer: public buffer
{
public:
  host_buffer(driver::context* context, size_t size);
};

// OpenCL
class ocl_buffer: public buffer
{
public:
  ocl_buffer(driver::context* context, size_t size);
};

// CUDA
class cu_buffer: public buffer
{
public:
  cu_buffer(driver::context* context, size_t size);
  cu_buffer(driver::context* context, CUdeviceptr cu, bool take_ownership);
  void set_zero(triton::driver::stream *queue, size_t size);
};

}
}

#endif
