#include "isaac/driver/event.h"
#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{

Event::Event(backend_type backend) : backend_(backend), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      cuda::check(dispatch::dispatch::cuEventCreate(&h_.cu().first, CU_EVENT_DEFAULT));
      cuda::check(dispatch::dispatch::cuEventCreate(&h_.cu().second, CU_EVENT_DEFAULT));
      break;
    case OPENCL:
      break;
    default:
      throw;
  }
}

Event::Event(cl_event const & event, bool take_ownership) : backend_(OPENCL), h_(backend_, take_ownership)
{
  h_.cl() = event;
}

long Event::elapsed_time() const
{
  switch(backend_)
  {
    case CUDA:
      float time;
      cuda::check(dispatch::cuEventElapsedTime(&time, h_.cu().first, h_.cu().second));
      return 1e6*time;
    case OPENCL:
      return static_cast<long>(ocl::info<CL_PROFILING_COMMAND_END>(h_.cl()) - ocl::info<CL_PROFILING_COMMAND_START>(h_.cl()));
    default:
      throw;
  }
}

HANDLE_TYPE(cl_event, cu_event_t) & Event::handle()
{ return h_; }

}

}
