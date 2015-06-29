#include "isaac/driver/context.h"
#include <iostream>

namespace isaac
{

namespace driver
{

Context::Context(cl::Context const & context) : backend_(OPENCL), device_(context.getInfo<CL_CONTEXT_DEVICES>()[0]), h_(backend_)
{
    *h_.cl = context;
}

Context::Context(Device const & device) : backend_(device.backend_), device_(device), h_(backend_)
{
#ifndef ANDROID
  if (std::getenv("ISAAC_CACHE_PATH"))
    cache_path_ = std::getenv("ISAAC_CACHE_PATH");
  else
#endif
    cache_path_ = "";

  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      cuda::check(cuCtxCreate(h_.cu.get(), CU_CTX_SCHED_AUTO, *device.h_.cu));
      break;
#endif
    case OPENCL:
      cl_int err;
      *h_.cl = cl::Context(std::vector<cl::Device>(1, *device_.h_.cl), NULL, NULL, NULL, &err);
      ocl::check(err);
      break;
    default:
      throw;
  }
}

bool Context::operator==(Context const & other) const
{ return h_==other.h_; }

bool Context::operator<(Context const & other) const
{ return h_<other.h_; }

Device const & Context::device() const
{ return device_; }

backend_type Context::backend() const
{ return backend_; }


}
}
