#include "isaac/driver/command_queue.h"
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/event.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/driver/buffer.h"
#include <iostream>
namespace isaac
{

namespace driver
{

CommandQueue::CommandQueue(cl::CommandQueue const & queue) : backend_(OPENCL), context_(queue.getInfo<CL_QUEUE_CONTEXT>()), device_(queue.getInfo<CL_QUEUE_DEVICE>()), h_(backend_)
{
  *h_.cl = queue;
}

CommandQueue::CommandQueue(Context const & context, Device const & device, cl_command_queue_properties properties): backend_(device.backend_), context_(context), device_(device), h_(backend_)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA: cuda::check(cuStreamCreate(h_.cu.get(), 0)); break;
#endif
    case OPENCL: *h_.cl = cl::CommandQueue(*context.h_.cl, *device.h_.cl, properties); break;
    default: throw;
  }
}

Context const & CommandQueue::context() const
{ return context_; }

Device const & CommandQueue::device() const
{ return device_; }

void CommandQueue::synchronize()
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA: cuda::check(cuStreamSynchronize(*h_.cu)); break;
#endif
    case OPENCL: h_.cl->finish(); break;
    default: throw;
  }
}

Event CommandQueue::enqueue(Kernel const & kernel, NDRange global, driver::NDRange local, std::vector<Event> const *)
{
  Event event(backend_);
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      cuda::check(cuEventRecord(event.h_.cu->first, *h_.cu));
      cuda::check(cuLaunchKernel(*kernel.h_.cu, global[0]/local[0], global[1]/local[1], global[2]/local[2],
                    local[0], local[1], local[2], 0, *h_.cu,(void**)&kernel.cu_params_[0], NULL));
      cuda::check(cuEventRecord(event.h_.cu->second, *h_.cu));
      break;
#endif
    case OPENCL:
      h_.cl->enqueueNDRangeKernel(*kernel.h_.cl, cl::NullRange, (cl::NDRange)global, (cl::NDRange)local, NULL, event.h_.cl.get());
      break;
    default: throw;
  }
  return event;
}

void CommandQueue::write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      if(blocking)
        cuda::check(cuMemcpyHtoD(*buffer.h_.cu + offset, ptr, size));
      else
        cuda::check(cuMemcpyHtoDAsync(*buffer.h_.cu + offset, ptr, size, *h_.cu));
      break;
#endif
    case OPENCL:
      h_.cl->enqueueWriteBuffer(*buffer.h_.cl, blocking, offset, size, ptr);
      break;
    default: throw;
  }
}

void CommandQueue::read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      if(blocking)
        cuda::check(cuMemcpyDtoH(ptr, *buffer.h_.cu + offset, size));
      else
        cuda::check(cuMemcpyDtoHAsync(ptr, *buffer.h_.cu + offset, size, *h_.cu));
      break;
#endif
    case OPENCL:
      h_.cl->enqueueReadBuffer(*buffer.h_.cl, blocking, offset, size, ptr);
      break;
    default: throw;
  }
}

bool CommandQueue::operator==(CommandQueue const & other) const
{ return h_ == other.h_; }

bool CommandQueue::operator<(CommandQueue const & other) const
{ return h_ < other.h_; }

HANDLE_TYPE(cl::CommandQueue, CUstream) & CommandQueue::handle()
{ return h_; }

}

}
