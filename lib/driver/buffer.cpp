#include "isaac/driver/buffer.h"
#include <iostream>

namespace isaac
{

namespace driver
{

Buffer::Buffer(cl::Buffer const & buffer) : backend_(OPENCL), context_(buffer.getInfo<CL_MEM_CONTEXT>()), h_(backend_)
{
  *h_.cl = buffer;
}

Buffer::Buffer(Context const & context, std::size_t size) : backend_(context.backend_), context_(context), h_(backend_)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA: cuda::check(cuMemAlloc(h_.cu.get(), size)); break;
#endif
    case OPENCL: *h_.cl = cl::Buffer(*context.h_.cl, CL_MEM_READ_WRITE, size); break;
    default: throw;
  }
}

Context const & Buffer::context() const
{ return context_; }

bool Buffer::operator==(Buffer const & other) const
{ return h_==other.h_; }

bool Buffer::operator<(Buffer const & other) const
{ return h_<other.h_; }

HANDLE_TYPE(cl::Buffer, CUdeviceptr) & Buffer::handle()
{ return h_; }

HANDLE_TYPE(cl::Buffer, CUdeviceptr) const & Buffer::handle() const
{ return h_; }

}

}
