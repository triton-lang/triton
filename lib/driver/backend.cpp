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

module& backend::modules::get(driver::stream const & stream, std::string const & name, std::string const & src){
  std::tuple<driver::stream, std::string> key(stream, name);
  if(cache_.find(key)==cache_.end())
    return *cache_.insert(std::make_pair(key, new module(stream.context(), src))).first->second;
  return *cache_.at(key);
}

std::map<std::tuple<stream, std::string>, module * >  backend::modules::cache_;

/*-----------------------------------*/
//-----------  Kernels --------------*/
/*-----------------------------------*/

void backend::kernels::release(){
  for(auto & x: cache_)
    delete x.second;
  cache_.clear();
}

kernel & backend::kernels::get(driver::module const & program, std::string const & name){
  std::tuple<module, std::string> key(program, name);
  if(cache_.find(key)==cache_.end())
    return *cache_.insert(std::make_pair(key, new kernel(program, name.c_str()))).first->second;
  return *cache_.at(key);
}

std::map<std::tuple<module, std::string>, kernel * > backend::kernels::cache_;

/*-----------------------------------*/
//------------  Queues --------------*/
/*-----------------------------------*/

void backend::streams::init(std::list<const context *> const & contexts){
  for(context const * ctx : contexts)
    if(cache_.find(*ctx)==cache_.end())
      cache_.insert(std::make_pair(*ctx, std::vector<stream*>{new stream(*ctx)}));
}

void backend::streams::release(){
  for(auto & x: cache_)
    for(auto & y: x.second)
      delete y;
  cache_.clear();
}

stream & backend::streams::get_default()
{ return get(contexts::get_default(), 0); }

stream & backend::streams::get(driver::context const & context, unsigned int id){
  init(std::list<driver::context const *>(1,&context));
  for(auto & x : cache_)
    if(x.first==context)
      return *x.second[id];
  throw;
}

void backend::streams::get(driver::context const & context, std::vector<stream*> & queues){
  init(std::list<driver::context const *>(1,&context));
  queues = cache_.at(context);
}

std::map<context, std::vector<stream*> > backend::streams::cache_;

/*-----------------------------------*/
//------------  Contexts ------------*/
/*-----------------------------------*/

void backend::contexts::init(std::vector<platform> const & platforms){
  for(platform const & platform: platforms){
    for(device const & device: platform.devices())
      cache_.push_back(new context(device));
  }
}

void backend::contexts::release(){
  for(auto & x: cache_)
    delete x;
  cache_.clear();
}

driver::context const & backend::contexts::get_default(){
  backend::init();
  std::list<context const *>::const_iterator it = cache_.begin();
  std::advance(it, default_device);
  return **it;
}

void backend::contexts::get(std::list<context const *> & contexts){
  backend::init();
  contexts = cache_;
}

std::list<context const *> backend::contexts::cache_;



/*-----------------------------------*/
//------------  General -------------*/
/*-----------------------------------*/

std::vector<device> backend::devices(){
  std::vector<platform> platforms = backend::platforms();
  std::vector<device> result;
  for(platform const & platform: platforms){
    auto devices = platform.devices();
    result.insert(result.end(), devices.begin(), devices.end());
  }
  return result;
}

std::vector<platform> backend::platforms(){
  std::vector<platform> platforms;
  //if CUDA is here
  if(dispatch::cuinit())
    platforms.push_back(platform());
  if(platforms.empty())
    throw std::runtime_error("ISAAC: No backend available. Make sure CUDA is available in your library path");
  return platforms;
}

void backend::synchronize(driver::context const & context){
  for(stream * queue: streams::cache_.at(context))
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
  std::vector<platform> platforms = backend::platforms();
  contexts::init(platforms);
  streams::init(contexts::cache_);
}

unsigned int backend::default_device = 0;

}

}
