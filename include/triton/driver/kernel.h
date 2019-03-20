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

#ifndef TDL_INCLUDE_DRIVER_KERNEL_H
#define TDL_INCLUDE_DRIVER_KERNEL_H

#include "triton/driver/module.h"
#include "triton/driver/handle.h"

#include <memory>

namespace triton
{

namespace driver
{

class cu_buffer;

// Base
class kernel: public polymorphic_resource<CUfunction, cl_kernel> {
public:
  kernel(driver::module* program, CUfunction fn, bool has_ownership);
  kernel(driver::module* program, cl_kernel fn, bool has_ownership);
  // Getters
  driver::module* module();
  // Factory methods
  static kernel* create(driver::module* program, const char* name);
  // Arguments setters
  virtual void setArg(unsigned int index, std::size_t size, void* ptr) = 0;
  virtual void setArg(unsigned int index, buffer *) = 0;
  template<class T> void setArg(unsigned int index, T value) { setArg(index, sizeof(T), (void*)&value); }
private:
  driver::module* program_;
};

// OpenCL
class ocl_kernel: public kernel {
public:
  //Constructors
  ocl_kernel(driver::module* program, const char* name);
  // Arguments setters
  void setArg(unsigned int index, std::size_t size, void* ptr);
  void setArg(unsigned int index, driver::buffer* buffer);

};

// CUDA
class cu_kernel: public kernel {
public:
  //Constructors
  cu_kernel(driver::module* program, const char * name);
  // Arguments setters
  void setArg(unsigned int index, std::size_t size, void* ptr);
  void setArg(unsigned int index, driver::buffer* buffer);
  //Arguments getters
  void* const* cu_params() const;

private:
  handle<CUfunction> cu_;
  driver::cu_module* program_;
  std::vector<std::shared_ptr<void> >  cu_params_store_;
  std::vector<void*>  cu_params_;
};

}

}

#endif

