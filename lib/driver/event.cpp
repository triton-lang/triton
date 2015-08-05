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
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      cuda::check(cuEventCreate(&h_.cu->first, CU_EVENT_DEFAULT));
      cuda::check(cuEventCreate(&h_.cu->second, CU_EVENT_DEFAULT));
      break;
#endif
    case OPENCL: break;
    default: throw;
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
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      float time;
      cuda::check(cuEventElapsedTime(&time, h_.cu->first, h_.cu->second));
      return 1e6*time;
#endif
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
