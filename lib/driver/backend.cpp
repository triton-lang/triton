/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "triton/driver/dispatch.h"
#include "triton/driver/backend.h"
#include "triton/driver/buffer.h"
#include "triton/driver/context.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"

#include <assert.h>
#include <stdexcept>
#include <vector>

namespace triton
{

namespace driver
{

/*-----------------------------------*/
//---------- Modules ----------------*/
/*-----------------------------------*/

void backend::modules::release(){
  for(auto & x: cache_)
    delete x.second;
  cache_.clear();
}

Module& backend::modules::get(Stream const & stream, std::string const & name, std::string const & src){
  std::tuple<Stream, std::string> key(stream, name);
  if(cache_.find(key)==cache_.end())
    return *cache_.insert(std::make_pair(key, new Module(stream.context(), src))).first->second;
  return *cache_.at(key);
}

std::map<std::tuple<Stream, std::string>, Module * >  backend::modules::cache_;

/*-----------------------------------*/
//-----------  Kernels --------------*/
/*-----------------------------------*/

void backend::kernels::release(){
  for(auto & x: cache_)
    delete x.second;
  cache_.clear();
}

Kernel & backend::kernels::get(Module const & program, std::string const & name){
  std::tuple<Module, std::string> key(program, name);
  if(cache_.find(key)==cache_.end())
    return *cache_.insert(std::make_pair(key, new Kernel(program, name.c_str()))).first->second;
  return *cache_.at(key);
}

std::map<std::tuple<Module, std::string>, Kernel * > backend::kernels::cache_;

/*-----------------------------------*/
//------------  Queues --------------*/
/*-----------------------------------*/

void backend::streams::init(std::list<const Context *> const & contexts){
  for(Context const * ctx : contexts)
    if(cache_.find(*ctx)==cache_.end())
      cache_.insert(std::make_pair(*ctx, std::vector<Stream*>{new Stream(*ctx)}));
}

void backend::streams::release(){
  for(auto & x: cache_)
    for(auto & y: x.second)
      delete y;
  cache_.clear();
}

Stream & backend::streams::get_default()
{ return get(contexts::get_default(), 0); }

Stream & backend::streams::get(Context const & context, unsigned int id){
  init(std::list<Context const *>(1,&context));
  for(auto & x : cache_)
    if(x.first==context)
      return *x.second[id];
  throw;
}

void backend::streams::get(Context const & context, std::vector<Stream*> & queues){
  init(std::list<Context const *>(1,&context));
  queues = cache_.at(context);
}

std::map<Context, std::vector<Stream*> > backend::streams::cache_;

/*-----------------------------------*/
//------------  Contexts ------------*/
/*-----------------------------------*/

void backend::contexts::init(std::vector<Platform> const & platforms){
  for(Platform const & platform: platforms){
    for(Device const & device: platform.devices())
      cache_.push_back(new Context(device));
  }
}

void backend::contexts::release(){
  for(auto & x: cache_)
    delete x;
  cache_.clear();
}

Context const & backend::contexts::get_default(){
  backend::init();
  std::list<Context const *>::const_iterator it = cache_.begin();
  std::advance(it, default_device);
  return **it;
}

void backend::contexts::get(std::list<Context const *> & contexts){
  backend::init();
  contexts = cache_;
}

std::list<Context const *> backend::contexts::cache_;



/*-----------------------------------*/
//------------  General -------------*/
/*-----------------------------------*/

std::vector<Device> backend::devices(){
  std::vector<Platform> platforms = backend::platforms();
  std::vector<Device> result;
  for(Platform const & platform: platforms){
    auto devices = platform.devices();
    result.insert(result.end(), devices.begin(), devices.end());
  }
  return result;
}

std::vector<Platform> backend::platforms(){
  std::vector<Platform> platforms;
  //if CUDA is here
  if(dispatch::cuinit())
    platforms.push_back(Platform());
  if(platforms.empty())
    throw std::runtime_error("ISAAC: No backend available. Make sure CUDA is available in your library path");
  return platforms;
}

void backend::synchronize(Context const & context){
  for(Stream * queue: streams::cache_.at(context))
    queue->synchronize();
}


void backend::release(){
  backend::kernels::release();
//  backend::programs::release();
  backend::streams::release();
  backend::contexts::release();
}


void backend::init(){
  if(!contexts::cache_.empty())
    return;
  std::vector<Platform> platforms = backend::platforms();
  contexts::init(platforms);
  streams::init(contexts::cache_);
}

unsigned int backend::default_device = 0;

}

}
