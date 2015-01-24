#include "atidlas/cl/queues.h"
#include <assert.h>
#include <stdexcept>

namespace atidlas
{

namespace cl
{

void synchronize(cl::Context const & context)
{
  std::vector<cl::CommandQueue> & q = get_queues(context);
  for(std::vector<cl::CommandQueue>::iterator it = q.begin() ; it != q.end() ; ++it)
    it->finish();
}

queues_t init_queues()
{
  queues_t result;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for(std::vector<cl::Platform>::iterator it = platforms.begin() ; it != platforms.end() ; ++it)
  {
    std::vector<cl::Device> devices;
    it->getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for(std::vector<cl::Device>::iterator itt = devices.begin() ; itt != devices.end() ; ++itt)
    {
      std::vector<cl::Device> current(1, *itt);
      cl::Context context(current);
      cl::CommandQueue queue(context, *itt);
      result.push_back(std::make_pair(context, std::vector<cl::CommandQueue>(1, queue)));
    }
  }

  return result;
}


cl::Context default_context()
{
  return queues[default_context_idx].second.front().getInfo<CL_QUEUE_CONTEXT>();
}

std::vector<cl::CommandQueue> & get_queues(cl::Context const & ctx)
{
  for(queues_t::iterator it = queues.begin() ; it != queues.end() ; ++it)
    if(it->first()==ctx())
      return it->second;
  throw std::out_of_range("The context provided is not registered");
}

cl::CommandQueue & get_queue(cl::Context const & ctx, std::size_t idx)
{ return get_queues(ctx)[idx]; }


unsigned int default_context_idx = 0;

queues_t queues = init_queues();
kernels_t kernels = kernels_t();

}

}
