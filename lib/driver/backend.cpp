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

void backend::init()
{
  if(contexts::contexts_.empty())
  {
      std::vector<Platform> platforms;
      backend::platforms(platforms);
      for(Platform const & platform: platforms)
      {
          std::vector<Device> devices;
          platform.devices(devices);
          for(Device const & device: devices){
              contexts::contexts_.push_back(new Context(device));
              queues::queues_.insert(std::make_pair(contexts::contexts_.back(), std::vector<CommandQueue*>{new CommandQueue(*contexts::contexts_.back(), device, queue_properties)}));
          }
      }
  }
}

CommandQueue & backend::queues::get(Context const & context, unsigned int id)
{
  assert(id < queues_.size());
  init();
  for(auto & x : queues_)
    if(x.first==&context)
        return *x.second[id];
  throw;
}

void backend::queues::get(Context const & context, std::vector<CommandQueue*> queues)
{
    queues = queues_[&context];
}


Context const & backend::contexts::import(cl_context context)
{
  for(driver::Context const * x: contexts_)
      if(x->handle().cl()==context)
          return *x;
  contexts_.emplace_back(new Context(context, false));
  return *contexts_.back();
}


Context const & backend::contexts::get_default()
{
  init();
  std::list<Context const *>::const_iterator it = contexts_.begin();
  std::advance(it, default_device);
  return **it;
}

void backend::contexts::get(std::list<Context const *> & contexts)
{
  init();
  contexts = contexts_;
}

void backend::contexts::release()
{
    for(auto & x: contexts_)
    {
        delete x;
        x = NULL;
    }
}

void backend::platforms(std::vector<Platform> & platforms)
{
  #ifdef ISAAC_WITH_CUDA
    platforms.push_back(Platform(CUDA));
  #endif
    cl_uint nplatforms;
    ocl::check(clGetPlatformIDs(0, NULL, &nplatforms));
    std::vector<cl_platform_id> clplatforms(nplatforms);
    ocl::check(clGetPlatformIDs(nplatforms, clplatforms.data(), NULL));
    for(cl_platform_id p: clplatforms)
        platforms.push_back(Platform(p));
}

void backend::synchronize(Context const & context)
{
    for(CommandQueue * queue: queues::queues_.at(&context))
        queue->synchronize();
}

void backend::queues::release()
{
    for(auto & x: queues_)
        for(auto & y: x.second)
        {
            delete y;
            y = NULL;
        }
}

void backend::release()
{
    backend::programs::release();
    backend::queues::release();
    backend::contexts::release();
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

std::list<Context const *> backend::contexts::contexts_;

std::map<Context const *, std::vector<CommandQueue*> > backend::queues::queues_;


}

}
