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

context::context(CUcontext cu, bool take_ownership):
  polymorphic_resource(cu, take_ownership),
  cache_path_(get_cache_path()) {
}

context::context(host_context_t hst, bool take_ownership):
  polymorphic_resource(hst, take_ownership),
  cache_path_(get_cache_path()){
}

context* context::create(driver::device *dev){
  switch(dev->backend()){
  case CUDA: return new cu_context  ((driver::cu_device*)dev  );
  case Host: return new host_context((driver::host_device*)dev);
  default: throw std::runtime_error("Unknown backend");
  }
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
//         Host             //
/* ------------------------ */

host_context::host_context(driver::host_device*): context(host_context_t(), true){

}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */
cu_context::cu_context(CUcontext context, bool take_ownership):
    driver::context(context, take_ownership)
{ }

cu_context::cu_context(driver::cu_device* dev): context(CUcontext(), true){
  dispatch::cuCtxCreate(&*cu_, CU_CTX_SCHED_AUTO, *dev->cu());
}

/* ------------------------ */
//         HIP              //
/* ------------------------ */
//hip_context::hip_context(CUcontext context, bool take_ownership):
//    driver::context(context, take_ownership)
//{ }

//hip_context::hip_context(driver::hip_device* dev): context(CUcontext(), true){
//  dispatch::cuCtxCreate(&*hip_, HIP_CTX_SCHED_AUTO, *dev->hip());
//}

}
}
