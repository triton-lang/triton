#include <iostream>
#include "isaac/driver/context.h"
#include "helpers/ocl/infos.hpp"
#include "isaac/driver/program.h"
namespace isaac
{

namespace driver
{

Context::Context(cl_context const & context, bool take_ownership) : backend_(OPENCL), device_(ocl::info<CL_CONTEXT_DEVICES>(context)[0], false), h_(backend_, take_ownership), programs_(*this)
{
    h_.cl() = context;

#ifndef ANDROID
  if (std::getenv("ISAAC_CACHE_PATH"))
    cache_path_ = std::getenv("ISAAC_CACHE_PATH");
  else
#endif
    cache_path_ = "";

}

Context::Context(Device const & device) : backend_(device.backend_), device_(device), h_(backend_, true), programs_(*this)
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


///////////////////////

Context::ProgramsHandler::ProgramsHandler(const Context &context) : context_(context){ }

Program const & Context::ProgramsHandler::add(std::string const & name, std::string const & src)
{
    std::map<std::string, Program>::iterator it = programs_.find(name);
    if(it==programs_.end())
    {
        std::string extensions;
        std::string ext = "cl_khr_fp64";
        if(context_.device().extensions().find(ext)!=std::string::npos)
          extensions = "#pragma OPENCL EXTENSION " + ext + " : enable\n";
        return programs_.insert(std::make_pair(name, driver::Program(context_, extensions + src))).first->second;
    }
    return it->second;
}

const Program * Context::ProgramsHandler::find(const std::string &name)
{
    std::map<std::string, Program>::const_iterator it = programs_.find(name);
    if(it==programs_.end())
        return NULL;
    return &it->second;
}

}
}
