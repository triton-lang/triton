#include "atidlas/cl/queues.h"
#include <assert.h>

namespace atidlas
{

namespace cl
{

void synchronize(cl::Context const & context)
{
  std::vector<cl::CommandQueue> & q = queues[context];
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
      result.insert(std::make_pair(context, std::vector<cl::CommandQueue>(1, queue)));
    }
  }

  return result;
}

cl::Context default_context()
{ return queues.begin()->second.front().getInfo<CL_QUEUE_CONTEXT>(); }

queues_t queues = init_queues();
kernels_t kernels = kernels_t();

}

}
