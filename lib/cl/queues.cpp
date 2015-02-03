#include "atidlas/cl/queues.h"
#include <assert.h>
#include <stdexcept>

namespace atidlas
{

namespace cl_ext
{

cl_command_queue_properties queue_properties = 0;
unsigned int default_context_idx = 0;
queues_t queues;
kernels_t kernels;

void synchronize(cl::Context const & context)
{
  std::vector<cl::CommandQueue> & q = get_queues(context);
  for(std::vector<cl::CommandQueue>::iterator it = q.begin() ; it != q.end() ; ++it)
    it->finish();
}

void init_queues()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for(std::vector<cl::Platform>::iterator it = platforms.begin() ; it != platforms.end() ; ++it)
    {
      std::vector<cl::Device> devices;
      it->getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for(std::vector<cl::Device>::iterator itt = devices.begin() ; itt != devices.end() ; ++itt)
        queues.push_back(std::make_pair(cl::Context(std::vector<cl::Device>(1, *itt)), std::vector<cl::CommandQueue>()));
    }
    for(queues_t::iterator it = queues.begin() ; it != queues.end() ; ++it)
      it->second.push_back(cl::CommandQueue(it->first, it->first.getInfo<CL_CONTEXT_DEVICES>()[0], queue_properties));
}

cl::Context default_context()
{
  if(queues.empty())
    init_queues();
  return queues.begin()->first;
}

std::vector<cl::CommandQueue> & get_queues(cl::Context const & ctx)
{
  if(queues.empty())
    init_queues();
  for(queues_t::iterator it = queues.begin() ; it != queues.end() ; ++it)
    if(it->first()==ctx()) return it->second;
  throw std::out_of_range("No such context registered in the backend. Please run atidlas::cl_ext:;register(context, queues)");
}

cl::CommandQueue & get_queue(cl::Context const & ctx, std::size_t idx)
{
  return get_queues(ctx)[idx];
}



}

}
