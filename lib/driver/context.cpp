#include <iostream>

#include "isaac/driver/context.h"
#include "isaac/driver/program.h"

#include "helpers/ocl/infos.hpp"
#include "getenv.hpp"

namespace isaac
{

namespace driver
{

Context::Context(cl_context const & context, bool take_ownership) : backend_(OPENCL), device_(ocl::info<CL_CONTEXT_DEVICES>(context)[0], false), cache_path_(tools::getenv("ISAAC_CACHE_PATH")), h_(backend_, take_ownership)
{
    h_.cl() = context;
}

Context::Context(Device const & device) : backend_(device.backend_), device_(device), cache_path_(tools::getenv("ISAAC_CACHE_PATH")), h_(backend_, true)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      cuda::check(cuCtxCreate(&h_.cu(), CU_CTX_SCHED_AUTO, device.h_.cu()));
      break;
#endif
    case OPENCL:
      cl_int err;
      h_.cl() = clCreateContext(NULL, 1, &device_.h_.cl(), NULL, NULL, &err);
      ocl::check(err);
      break;
    default:
      throw;
  }
}

bool Context::operator==(Context const & other) const
{
    return h_==other.h_;
}

bool Context::operator<(Context const & other) const
{
    return h_<other.h_;
}

Device const & Context::device() const
{ return device_; }

backend_type Context::backend() const
{ return backend_; }

}
}
