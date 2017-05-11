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

#include <iostream>
#include <cassert>

#include "isaac/driver/context.h"
#include "isaac/driver/module.h"

#include "isaac/tools/sys/getenv.hpp"
#include "isaac/tools/sys/mkdir.hpp"

namespace isaac
{

namespace driver
{

std::string Context::get_cache_path(){
  //user-specified cache path
  std::string result = tools::getenv("ISAAC_CACHE_PATH");
  if(!result.empty()){
    if(tools::mkpath(result)==0)
      return result;
  }
  //create in home
  result = tools::getenv("HOME");
  if(!result.empty())
  {
    result = result + "/.isaac/cache/";
    if(tools::mkpath(result)==0)
      return result;
  }
  //couldn't find a directory
  return "";
}

CUdevice Context::device(CUcontext){
  CUdevice res;
  dispatch::cuCtxGetDevice(&res);
  return res;
}

Context::Context(CUcontext context, bool take_ownership): cu_(context, take_ownership), device_(device(context), false), cache_path_(get_cache_path())
{ }

Context::Context(Device const & device): device_(device), cache_path_(get_cache_path())
{
  dispatch::cuCtxCreate(&*cu_, CU_CTX_SCHED_AUTO, (CUdevice)device);
  dispatch::cuCtxPopCurrent_v2(NULL);
}

Device const & Context::device() const
{ return device_; }

std::string const & Context::cache_path() const
{ return cache_path_; }

Handle<CUcontext> const & Context::cu() const
{ return cu_; }

/* Context Switcher */
ContextSwitcher::ContextSwitcher(Context const & ctx): ctx_(ctx)
{
  dispatch::cuCtxPushCurrent_v2(ctx_);
}

ContextSwitcher::~ContextSwitcher()
{
  CUcontext tmp;
  dispatch::cuCtxPopCurrent_v2(&tmp);
  assert(tmp==(CUcontext)ctx_ && "Switching back to invalid context!");
}



}
}
