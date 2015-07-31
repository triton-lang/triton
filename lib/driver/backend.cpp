#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/command_queue.h"

#include <assert.h>
#include <stdexcept>
#include <vector>

namespace isaac
{

namespace driver
{

void backend::cuinit()
{
#ifdef ISAAC_WITH_CUDA
  cuda::check(cuInit(0));
  int N;
  cuda::check(cuDeviceGetCount(&N));
  for(int i = 0 ; i < N ; ++i)
  {
      Device device(i);
      contexts_.emplace_back(device);
      queues_.insert(std::make_pair(&contexts_.back(), std::vector<CommandQueue>{CommandQueue(contexts_.back(), device, queue_properties)}));
  }
#endif
}

void backend::clinit()
{
  cl_uint nplatforms;
  ocl::check(clGetPlatformIDs(0, NULL, &nplatforms));
  std::vector<cl_platform_id> platforms(nplatforms);
  ocl::check(clGetPlatformIDs(nplatforms, platforms.data(), NULL));
  for(cl_platform_id p : platforms)
  {
    cl_uint ndevices;
    ocl::check(clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices));
    std::vector<cl_device_id> devices(ndevices);
    ocl::check(clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndevices, devices.data(), NULL));
    for(cl_device_id d : devices){
      Device device(d);
      contexts_.emplace_back(device);
      queues_.insert(std::make_pair(&contexts_.back(), std::vector<CommandQueue>{CommandQueue(contexts_.back(), device, queue_properties)}));
    }
  }
}

void backend::init()
{
  if(contexts_.empty())
  {
    cuinit();
    clinit();
  }
}

std::vector<CommandQueue> & backend::queues(Context const & context)
{
  init();
  for(auto & x : queues_)
    if(x.first==&context)
        return x.second;
  throw;
}

Context const & backend::import(cl_context context)
{
  for(driver::Context const & x: contexts_)
      if(x.handle().cl()==context)
          return x;
  contexts_.emplace_back(context, false);
  return contexts_.back();
}


Context const & backend::default_context()
{
  init();
  std::list<Context>::const_iterator it = contexts_.begin();
  std::advance(it, default_device);
  return *it;
}

const std::list<Context> &backend::contexts()
{
  init();
  return contexts_;
}


ISAACAPI void synchronize(std::vector<CommandQueue > & queues)
{
  for(CommandQueue & q: queues)
    q.synchronize();
}

ISAACAPI void synchronize(Context const & context)
{ synchronize(backend::queues(context)); }


//Static variables

unsigned int backend::default_device = 0;

cl_command_queue_properties backend::queue_properties = 0;

std::list<Context> backend::contexts_;

std::map<Context*, std::vector<CommandQueue>> backend::queues_;

}

}
