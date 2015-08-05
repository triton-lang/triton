#include <iostream>
#include "isaac/driver/context.h"
#include "helpers/ocl/infos.hpp"
#include "isaac/driver/program.h"
#include "isaac/tools/getenv.hpp"

namespace isaac
{

namespace driver
{

void Context::init_cache_path()
{
#ifndef ANDROID
  #ifdef _MSC_VER
      char* cache_path = 0;
      std::size_t sz = 0;
      _dupenv_s(&cache_path, &sz, "ISAAC_CACHE_PATH");
  #else
      const char * cache_path = std::getenv("ISAAC_CACHE_PATH");
  #endif
  if (cache_path)
    cache_path_ = cache_path;
  else
#endif
    cache_path_ = "";
}

Context::Context(cl_context const & context, bool take_ownership) : backend_(OPENCL), device_(ocl::info<CL_CONTEXT_DEVICES>(context)[0], false), cache_path_(tools::getenv("ISAAC_CACHE_PATH")), h_(backend_, take_ownership)
{
    init_cache_path();
    h_.cl() = context;
}

Context::Context(Device const & device) : backend_(device.backend_), device_(device), cache_path_(tools::getenv("ISAAC_CACHE_PATH")), h_(backend_, true)
{
  init_cache_path();
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
      cuda::check(cuCtxCreate(h_.cu.get(), CU_CTX_SCHED_AUTO, *device.h_.cu));
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
