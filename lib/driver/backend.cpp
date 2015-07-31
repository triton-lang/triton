#include "isaac/driver/backend.h"
#include <assert.h>
#include <stdexcept>
#include <vector>

namespace isaac
{

namespace driver
{

std::vector<CommandQueue> & backend::append(Context const & context)
{
  data_.push_back(std::make_pair(context, std::vector<CommandQueue>(1, CommandQueue(context, context.device(), queue_properties))));
  return data_.back().second;
}

void backend::cuinit()
{
#ifdef ISAAC_WITH_CUDA
  cuda::check(cuInit(0));
  int N;
  cuda::check(cuDeviceGetCount(&N));
  for(int i = 0 ; i < N ; ++i)
    append(Context(Device(i));
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
    for(cl_device_id d : devices)
      append(Context(Device(d)));
  }
}

void backend::init()
{
  if(data_.empty())
  {
    cuinit();
    clinit();
  }
}

std::vector<CommandQueue> & backend::queues(Context const & context)
{
  init();
  for(auto & x : data_)
    if(x.first==context) return x.second;
  return append(context);
}

Context backend::default_context()
{
  init();
  container_type::iterator it = data_.begin();
  std::advance(it, default_device);
  return it->first;
}

std::vector<CommandQueue> & backend::default_queues()
{ return backend::queues(default_context()); }


backend::container_type const & backend::contexts()
{
  init();
  return data_;
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

backend::container_type backend::data_ = backend::container_type();

}

}
