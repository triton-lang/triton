#include "isaac/driver/event.h"

namespace isaac
{

namespace driver
{

Event::Event(backend_type backend) : backend_(backend), h_(backend_)
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
      return (h_.cl->getProfilingInfo<CL_PROFILING_COMMAND_END>() - h_.cl->getProfilingInfo<CL_PROFILING_COMMAND_START>());
    default:
      throw;
  }
}

}

}
