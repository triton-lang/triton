#include "isaac/driver/backend.h"
#include <assert.h>
#include <stdexcept>

namespace isaac
{

namespace driver
{

queues_type::queues_type(): default_device(0), queue_properties(0)
{}

std::vector<CommandQueue> & queues_type::append(Context const & context)
{
  data_.push_back(std::make_pair(context, std::vector<CommandQueue>(1, CommandQueue(context, context.device(), queue_properties))));
  return data_.back().second;
}

void queues_type::cuinit()
{
#ifdef ISAAC_WITH_CUDA
  cuda::check(cuInit(0));
  int N;
  cuda::check(cuDeviceGetCount(&N));
  for(int i = 0 ; i < N ; ++i)
    append(Context(Device(i));
#endif
}

void queues_type::clinit()
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for(auto & p : platforms)
  {
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for(auto & d : devices)
      append(Context(Device(d)));
  }
}

void queues_type::init()
{
  if(data_.empty())
  {
    cuinit();
    clinit();
  }
}

std::vector<CommandQueue> & queues_type::operator[](Context const & context)
{
  init();
  for(auto & x : data_)
    if(x.first==context) return x.second;
  return append(context);
}

Context queues_type::default_context()
{
  init();
  container_type::iterator it = data_.begin();
  std::advance(it, default_device);
  return it->first;
}

std::vector<CommandQueue> & queues_type::default_queues()
{ return (*this)[default_context()]; }


queues_type::container_type const & queues_type::contexts()
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
{ synchronize(queues[context]); }

ISAACAPI queues_type queues;

}

}
