#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/program_cache.h"

#include <assert.h>
#include <stdexcept>
#include <vector>

namespace isaac
{

namespace driver
{

/*-----------------------------------*/
//----------  Programs --------------*/
/*-----------------------------------*/

void backend::programs::release()
{
    for(auto & x: cache_)
        delete x.second;
    cache_.clear();
}

ProgramCache & backend::programs::get(CommandQueue const & queue, expression_type expression, numeric_type dtype)
{
    std::tuple<CommandQueue, expression_type, numeric_type> key(queue, expression, dtype);
    if(cache_.find(key)==cache_.end())
        return *cache_.insert(std::make_pair(key, new ProgramCache())).first->second;
    return *cache_.at(key);
}

std::map<std::tuple<CommandQueue, expression_type, numeric_type>, ProgramCache * >  backend::programs::cache_;

/*-----------------------------------*/
//------------  Queues --------------*/
/*-----------------------------------*/

void backend::queues::init(std::list<const Context *> const & contexts)
{
    for(Context const * ctx : contexts)
        if(cache_.find(*ctx)==cache_.end())
        cache_.insert(std::make_pair(*ctx, std::vector<CommandQueue*>{new CommandQueue(*ctx, ctx->device(), queue_properties)}));
}

void backend::queues::release()
{
    for(auto & x: cache_)
        for(auto & y: x.second)
            delete y;
    cache_.clear();
}


CommandQueue & backend::queues::get(Context const & context, unsigned int id)
{
  init(std::list<Context const *>(1,&context));
  for(auto & x : cache_)
    if(x.first==context)
        return *x.second[id];
  throw;
}

void backend::queues::get(Context const & context, std::vector<CommandQueue*> & queues)
{
    init(std::list<Context const *>(1,&context));
    queues = cache_.at(context);
}

std::map<Context, std::vector<CommandQueue*> > backend::queues::cache_;

/*-----------------------------------*/
//------------  Contexts ------------*/
/*-----------------------------------*/

void backend::contexts::init(std::vector<Platform> const & platforms)
{
    for(Platform const & platform: platforms)
    {
        std::vector<Device> devices;
        platform.devices(devices);
        for(Device const & device: devices)
            cache_.push_back(new Context(device));
    }
}

void backend::contexts::release()
{
    for(auto & x: cache_)
        delete x;
    cache_.clear();
}

Context const & backend::contexts::import(cl_context context)
{
  for(driver::Context const * x: cache_)
      if(x->handle().cl()==context)
          return *x;
  cache_.emplace_back(new Context(context, false));
  return *cache_.back();
}


Context const & backend::contexts::get_default()
{
  backend::init();
  std::list<Context const *>::const_iterator it = cache_.begin();
  std::advance(it, default_device);
  return **it;
}

void backend::contexts::get(std::list<Context const *> & contexts)
{
  backend::init();
  contexts = cache_;
}

std::list<Context const *> backend::contexts::cache_;



/*-----------------------------------*/
//------------  General -------------*/
/*-----------------------------------*/

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
    for(CommandQueue * queue: queues::cache_.at(context))
        queue->synchronize();
}


void backend::release()
{
    backend::programs::release();
    backend::queues::release();
    backend::contexts::release();
}


void backend::init()
{
  if(!contexts::cache_.empty())
      return;
  std::vector<Platform> platforms;
  backend::platforms(platforms);
  contexts::init(platforms);
  queues::init(contexts::cache_);
}

unsigned int backend::default_device = 0;

cl_command_queue_properties backend::queue_properties = 0;


}

}
