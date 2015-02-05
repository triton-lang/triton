#include "atidlas/cl_ext/backend.h"
#include <assert.h>
#include <stdexcept>

namespace atidlas
{

namespace cl_ext
{

void synchronize(cl::Context const & context)
{
  for(std::vector<cl::CommandQueue>::const_iterator it = queues[context].begin() ; it != queues[context].end() ; ++it)
    it->finish();
}

void queues_type::append(const cl::Context & context)
{
  data_.push_back(std::make_pair(context, std::vector<cl::CommandQueue>()));
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  for(auto & device : devices)
    data_.back().second.push_back(cl::CommandQueue(context, device, queue_properties));
}

void queues_type::init()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for(auto & platform : platforms)
    {
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for(auto & device : devices)
        data_.push_back(std::make_pair(cl::Context(std::vector<cl::Device>(1, device)), std::vector<cl::CommandQueue>()));
    }
    for(auto & elem : data_)
      elem.second.push_back(cl::CommandQueue(elem.first, elem.first.getInfo<CL_CONTEXT_DEVICES>()[0], queue_properties));
}

std::vector<cl::CommandQueue> & queues_type::operator [](cl::Context const & ctx)
{
  if(data_.empty())
    init();
  for(auto & elem : data_)
    if(elem.first()==ctx()) return elem.second;
  append(ctx);
  return data_.back().second;
}

cl::Context queues_type::default_context()
{
  if(data_.empty())
    init();
  data_type::iterator it = data_.begin();
  std::advance(it, default_context_idx);
  return it->first;
}

cl::Context default_context()
{
  return queues.default_context();
}

queues_type::data_type const & queues_type::data()
{
  if(data_.empty())
    init();
  return data_;
}

cl_command_queue_properties queue_properties = 0;
unsigned int default_context_idx = 0;
queues_type queues;
kernels_type kernels;

}

}
