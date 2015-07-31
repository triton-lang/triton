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
      contexts_.emplace_back(new Context(device));
      queues_.insert(std::make_pair(contexts_.back(), std::vector<CommandQueue*>{new CommandQueue(contexts_.back(), device, queue_properties)}));
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
      contexts_.emplace_back(new Context(device));
      queues_.insert(std::make_pair(contexts_.back(), std::vector<CommandQueue*>{new CommandQueue(*contexts_.back(), device, queue_properties)}));
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

CommandQueue & backend::queue(Context const & context, unsigned int id)
{
  assert(id < queues_.size());
  init();
  for(auto & x : queues_)
    if(x.first==&context)
        return *x.second[id];
  throw;
}

Context const & backend::import(cl_context context)
{
  for(driver::Context const * x: contexts_)
      if(x->handle().cl()==context)
          return *x;
  contexts_.emplace_back(new Context(context, false));
  return *contexts_.back();
}


Context const & backend::default_context()
{
  init();
  std::list<Context const *>::const_iterator it = contexts_.begin();
  std::advance(it, default_device);
  return **it;
}

const std::list<Context const *> &backend::contexts()
{
  init();
  return contexts_;
}


void backend::synchronize(Context const & context)
{
    for(CommandQueue * queue: queues_.at(&context))
        queue->synchronize();
}

void backend::release()
{
    //programs
    backend::programs::release();

    //queues
    for(auto & x: queues_)
        for(auto & y: x.second)
        {
            delete y;
            y = NULL;
        }

    //context
    for(auto & x: contexts_)
    {
        delete x;
        x = NULL;
    }
}

/* ---- Programs -----*/

Program const & backend::programs::add(Context const & context, std::string const & name, std::string const & src)
{
    std::map<std::string, Program*> & pgms = programs_.at(&context);
    std::map<std::string, Program*>::iterator it = pgms.find(name);
    if(it==pgms.end())
    {
        std::string extensions;
        std::string ext = "cl_khr_fp64";
        if(context.device().extensions().find(ext)!=std::string::npos)
          extensions = "#pragma OPENCL EXTENSION " + ext + " : enable\n";
        return *pgms.insert(std::make_pair(name, new driver::Program(context, extensions + src))).first->second;
    }
    return *it->second;
}

const Program * backend::programs::find(Context const & context, const std::string &name)
{
    std::map<std::string, Program*> & pgms = programs_[&context];
    std::map<std::string, Program*>::const_iterator it = pgms.find(name);
    if(it==pgms.end())
        return NULL;
    return it->second;
}

void backend::programs::release()
{
    for(auto & x: programs_)
        for(auto & y: x.second)
        {
            delete y.second;
            y.second = NULL;
        }
}

std::map<driver::Context const *, std::map<std::string, Program*> > backend::programs::programs_;


//Static variables

unsigned int backend::default_device = 0;

cl_command_queue_properties backend::queue_properties = 0;

std::list<Context const *> backend::contexts_;

std::map<Context const *, std::vector<CommandQueue*> > backend::queues_;


}

}
