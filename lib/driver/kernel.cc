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

#include <string.h>
#include "triton/driver/kernel.h"
#include "triton/driver/buffer.h"

namespace triton
{

namespace driver
{


/* ------------------------ */
//         Base             //
/* ------------------------ */

kernel::kernel(driver::module *program, CUfunction fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}


kernel::kernel(driver::module *program, host_function_t fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel* kernel::create(driver::module* program, const char* name) {
    switch(program->backend()){
    case CUDA: return new cu_kernel(program, name);
    case Host: return new host_kernel(program, name);
    default: throw std::runtime_error("unknown backend");
    }
}

driver::module* kernel::module() {
  return program_;
}

/* ------------------------ */
//         Host             //
/* ------------------------ */

host_kernel::host_kernel(driver::module* program, const char *name): kernel(program, host_function_t(), true) {
  hst_->fn = program->hst()->functions.at(name);
}

void host_kernel::setArg(unsigned int index, std::size_t size, void* ptr){
  if(index + 1> params_store_.size()){
    params_store_.resize(index+1);
    params_.resize(index+1);
  }
  params_store_[index].reset(malloc(size), free);
  memcpy(params_store_[index].get(), ptr, size);
  params_[index] = params_store_[index].get();
}

void host_kernel::setArg(unsigned int index, driver::buffer* buffer){
  if(buffer)
    kernel::setArg(index, (void*)buffer->hst()->data);
  else
    kernel::setArg(index, (std::ptrdiff_t)0);
}

const std::vector<void *> &host_kernel::params(){
  return params_;
}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */

cu_kernel::cu_kernel(driver::module *program, const char * name) : kernel(program, CUfunction(), true) {
  cu_params_store_.reserve(64);
  cu_params_.reserve(64);
  dispatch::cuModuleGetFunction(&*cu_, *program->cu(), name);
//  dispatch::cuFuncSetCacheConfig(*cu_, CU_FUNC_CACHE_PREFER_SHARED);
}

void cu_kernel::setArg(unsigned int index, std::size_t size, void* ptr){
  if(index + 1> cu_params_store_.size()){
    cu_params_store_.resize(index+1);
    cu_params_.resize(index+1);
  }
  cu_params_store_[index].reset(malloc(size), free);
  memcpy(cu_params_store_[index].get(), ptr, size);
  cu_params_[index] = cu_params_store_[index].get();
}

void cu_kernel::setArg(unsigned int index, driver::buffer* data){
  if(data)
    kernel::setArg(index, *data->cu());
  else
    kernel::setArg(index, (std::ptrdiff_t)0);
}

void* const* cu_kernel::cu_params() const
{ return cu_params_.data(); }


}

}

