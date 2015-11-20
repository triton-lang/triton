#include <iostream>
#include "isaac/driver/buffer.h"
#include "isaac/driver/backend.h"
#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{

Buffer::Buffer(CUdeviceptr h, bool take_ownership) : backend_(CUDA), context_(backend::contexts::get_default()), h_(backend_, take_ownership)
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
      cuda::check(dispatch::cuMemAlloc(&h_.cu(), size));
      break;
    case OPENCL:
      cl_int err;
      h_.cl() = dispatch::clCreateBuffer(context.h_.cl(), CL_MEM_READ_WRITE, size, NULL, &err);
      ocl::check(err);
      break;
    default:
      throw;
  }
}

Context const & Buffer::context() const
{ return context_; }

bool Buffer::operator==(Buffer const & other) const
{ return h_==other.h_; }

bool Buffer::operator<(Buffer const & other) const
{ return h_<other.h_; }

HANDLE_TYPE(cl_mem, CUdeviceptr) & Buffer::handle()
{ return h_; }

HANDLE_TYPE(cl_mem, CUdeviceptr) const & Buffer::handle() const
{ return h_; }

}

}
