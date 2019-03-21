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
#include "triton/driver/stream.h"
#include "triton/driver/buffer.h"
#include "triton/driver/context.h"
#include "triton/driver/dispatch.h"


namespace triton
{

namespace driver
{


//

buffer::buffer(driver::context* ctx, CUdeviceptr cu, bool take_ownership)
  : polymorphic_resource(cu, take_ownership), context_(ctx) { }

buffer::buffer(driver::context* ctx, cl_mem cl, bool take_ownership)
  : polymorphic_resource(cl, take_ownership), context_(ctx) { }

driver::context* buffer::context() {
  return context_;
}

buffer* buffer::create(driver::context* ctx, size_t size) {
  if(dynamic_cast<driver::cu_context*>(ctx))
    return new cu_buffer(ctx, size);
  if(dynamic_cast<driver::ocl_context*>(ctx))
    return new ocl_buffer(ctx, size);
  throw std::runtime_error("unknown context");
}

//

ocl_buffer::ocl_buffer(driver::context* context, size_t size)
  : buffer(context, cl_mem(), true){
  cl_int err;
  dispatch::clCreateBuffer(*context->cl(), CL_MEM_READ_WRITE, size, NULL, &err);
}


//

cu_buffer::cu_buffer(driver::context* context, size_t size)
  : buffer(context, CUdeviceptr(), true) {
  cu_context::context_switcher ctx_switch(*context_);
  dispatch::cuMemAlloc(&*cu_, size);
}

cu_buffer::cu_buffer(driver::context* context, CUdeviceptr cu, bool take_ownership)
  : buffer(context, cu, take_ownership){
}

void cu_buffer::set_zero(cu_stream const & queue, size_t size)
{
  cu_context::context_switcher ctx_switch(*context_);
  dispatch::cuMemsetD8Async(*cu_, 0, size, *queue.cu());
}

}

}
