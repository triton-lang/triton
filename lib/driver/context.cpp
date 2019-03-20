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

/* ------------------------ */
//         BASE             //
/* ------------------------ */

context::context(driver::device *dev, CUcontext cu, bool take_ownership):
  polymorphic_resource(cu, take_ownership),
  dev_(dev), cache_path_(get_cache_path()) {
}

context::context(driver::device *dev, cl_context cl, bool take_ownership):
  polymorphic_resource(cl, take_ownership),
  dev_(dev), cache_path_(get_cache_path()){

}

context* context::create(driver::device *dev){
  if(dynamic_cast<driver::cu_device*>(dev))
    return new cu_context(dev);
  if(dynamic_cast<driver::ocl_device*>(dev))
    return new ocl_context(dev);
  throw std::runtime_error("unknown context");
}


driver::device* context::device() const {
  return dev_;
}

std::string context::get_cache_path(){
  //user-specified cache path
  std::string result = tools::getenv("TRITON_CACHE_PATH");
  if(!result.empty()){
    if(tools::mkpath(result)==0)
      return result;
  }
  //create in home
  result = tools::getenv("HOME");
  if(!result.empty())
  {
    result = result + "/.triton/cache/";
    if(tools::mkpath(result)==0)
      return result;
  }
  //couldn't find a directory
  return "";
}

std::string const & context::cache_path() const{
  return cache_path_;
}


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

// RAII context switcher
cu_context::context_switcher::context_switcher(const context &ctx): ctx_((const cu_context&)ctx) {
  dispatch::cuCtxPushCurrent_v2(*ctx_.cu());
}

cu_context::context_switcher::~context_switcher() {
  CUcontext tmp;
  dispatch::cuCtxPopCurrent_v2(&tmp);
  assert(tmp==(CUcontext)ctx_ && "Switching back to invalid context!");
}

// import CUdevice
CUdevice cu_context::get_device_of(CUcontext context){
  dispatch::cuCtxPushCurrent_v2(context);
  CUdevice res;
  dispatch::cuCtxGetDevice(&res);
  dispatch::cuCtxPopCurrent_v2(NULL);
  return res;
}

// wrapper for cuda context
cu_context::cu_context(CUcontext context, bool take_ownership): driver::context(new driver::cu_device(get_device_of(context), false),
                                                                                context, take_ownership) {
}

cu_context::cu_context(driver::device* device): context(device, CUcontext(), true){
  dispatch::cuCtxCreate(&*cu_, CU_CTX_SCHED_AUTO, *((driver::cu_device*)dev_)->cu());
  dispatch::cuCtxPopCurrent_v2(NULL);
}


/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

ocl_context::ocl_context(driver::device* dev): context(dev, cl_context(), true) {
  cl_int err;
  *cl_ = dispatch::clCreateContext(nullptr, 1, &*dev->cl(), nullptr, nullptr, &err);
}



}
}
