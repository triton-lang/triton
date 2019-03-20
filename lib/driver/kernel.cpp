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
#include <cstring>

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

kernel::kernel(driver::module *program, cl_kernel fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel* kernel::create(driver::module* program, const char* name) {
  if(dynamic_cast<driver::cu_module*>(program))
    return new cu_kernel(program, name);
  if(dynamic_cast<driver::ocl_module*>(program))
    return new ocl_kernel(program, name);
  throw std::runtime_error("unknown program");
}

driver::module* kernel::module() {
  return program_;
}

/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

ocl_kernel::ocl_kernel(driver::module* program, const char* name): kernel(program, cl_kernel(), true) {
  cl_int err;
  *cl_ = dispatch::clCreateKernel(*program->cl(), name, &err);
}

void ocl_kernel::setArg(unsigned int index, std::size_t size, void* ptr) {
  dispatch::clSetKernelArg(*cl_, index, size, ptr);
}

void ocl_kernel::setArg(unsigned int index, driver::buffer* buffer) {
  dispatch::clSetKernelArg(*cl_, index, sizeof(cl_mem), (void*)&*buffer->cl());
}


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

cu_kernel::cu_kernel(driver::module *program, const char * name) : kernel(program, CUfunction(), true) {
  cu_params_store_.reserve(64);
  cu_params_.reserve(64);
  dispatch::cuModuleGetFunction(&*cu_, *program->cu(), name);
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

void cu_kernel::setArg(unsigned int index, driver::buffer* data)
{ return kernel::setArg(index, *data->cu());}

void* const* cu_kernel::cu_params() const
{ return cu_params_.data(); }


}

}

