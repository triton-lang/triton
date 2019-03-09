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

#include "triton/driver/context.h"
#include "triton/driver/module.h"

#include "triton/tools/sys/getenv.hpp"
#include "triton/tools/sys/mkdir.hpp"

namespace triton
{

namespace driver
{

std::string context::get_cache_path(){
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

CUdevice context::device(CUcontext context){
  dispatch::cuCtxPushCurrent_v2(context);
  CUdevice res;
  dispatch::cuCtxGetDevice(&res);
  dispatch::cuCtxPopCurrent_v2(NULL);
  return res;
}

context::context(CUcontext context, bool take_ownership): cu_(context, take_ownership), dvc_(device(context), false), cache_path_(get_cache_path())
{ }

context::context(driver::device const & device): dvc_(device), cache_path_(get_cache_path())
{
  dispatch::cuCtxCreate(&*cu_, CU_CTX_SCHED_AUTO, (CUdevice)device);
  dispatch::cuCtxPopCurrent_v2(NULL);
}

device const & context::device() const
{ return dvc_; }

std::string const & context::cache_path() const
{ return cache_path_; }

handle<CUcontext> const & context::cu() const
{ return cu_; }

/* Context Switcher */
ContextSwitcher::ContextSwitcher(driver::context const & ctx): ctx_(ctx)
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
