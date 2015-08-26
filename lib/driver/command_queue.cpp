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
      cuda::check(dispatch::cuStreamCreate(&h_.cu(), 0));
      break;

    case OPENCL:
    {
      cl_int err;
      h_.cl() = dispatch::clCreateCommandQueue(context.h_.cl(), device.h_.cl(), properties, &err);
      ocl::check(err);
      break;
    }
    default: throw;
  }
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
    case CUDA: cuda::check(dispatch::cuStreamSynchronize(h_.cu())); break;
    case OPENCL: ocl::check(dispatch::clFinish(h_.cl())); break;
    default: throw;
  }
}

Event CommandQueue::enqueue(Kernel const & kernel, NDRange global, driver::NDRange local, std::vector<Event> const *)
{
  Event event(backend_);
  switch(backend_)
  {
    case CUDA:
      cuda::check(dispatch::cuEventRecord(event.h_.cu().first, h_.cu()));
      cuda::check(dispatch::cuLaunchKernel(kernel.h_.cu(), global[0]/local[0], global[1]/local[1], global[2]/local[2],
                    local[0], local[1], local[2], 0, h_.cu(),(void**)&kernel.cu_params_[0], NULL));
      cuda::check(dispatch::cuEventRecord(event.h_.cu().second, h_.cu()));
      break;
    case OPENCL:
      ocl::check(dispatch::clEnqueueNDRangeKernel(h_.cl(), kernel.h_.cl(), global.dimension(), NULL, (const size_t *)global, (const size_t *) local, 0, NULL, &event.handle().cl()));
      break;
    default: throw;
  }
  return event;
}

void CommandQueue::write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr)
{
  switch(backend_)
  {
    case CUDA:
      if(blocking)
        cuda::check(dispatch::cuMemcpyHtoD(buffer.h_.cu() + offset, ptr, size));
      else
        cuda::check(dispatch::cuMemcpyHtoDAsync(buffer.h_.cu() + offset, ptr, size, h_.cu()));
      break;
    case OPENCL:
      ocl::check(dispatch::clEnqueueWriteBuffer(h_.cl(), buffer.h_.cl(), blocking?CL_TRUE:CL_FALSE, offset, size, ptr, 0, NULL, NULL));
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
        cuda::check(dispatch::cuMemcpyDtoH(ptr, buffer.h_.cu() + offset, size));
      else
        cuda::check(dispatch::cuMemcpyDtoHAsync(ptr, buffer.h_.cu() + offset, size, h_.cu()));
      break;
    case OPENCL:
      ocl::check(dispatch::clEnqueueReadBuffer(h_.cl(), buffer.h_.cl(), blocking?CL_TRUE:CL_FALSE, offset, size, ptr, 0, NULL, NULL));
      break;
    default: throw;
  }
}

bool CommandQueue::operator==(CommandQueue const & other) const
{ return h_ == other.h_; }

bool CommandQueue::operator<(CommandQueue const & other) const
{ return h_ < other.h_; }

HANDLE_TYPE(cl_command_queue, CUstream) & CommandQueue::handle()
{ return h_; }

}

}
