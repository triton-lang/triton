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

#ifndef TDL_INCLUDE_DRIVER_CONTEXT_H
#define TDL_INCLUDE_DRIVER_CONTEXT_H

#include "triton/driver/device.h"
#include "triton/driver/handle.h"

namespace triton
{
namespace driver
{

class context: public polymorphic_resource<CUcontext, cl_context, host_context_t>{
protected:
  static std::string get_cache_path();

public:
  context(driver::device *dev, CUcontext cu, bool take_ownership);
  context(driver::device *dev, cl_context cl, bool take_ownership);
  context(driver::device *dev, host_context_t hst, bool take_ownership);
  driver::device* device() const;
  std::string const & cache_path() const;
  // factory methods
  static context* create(driver::device *dev);

protected:
  driver::device* dev_;
  std::string cache_path_;
};

// Host
class host_context: public context {
public:
  host_context(driver::device* dev);
};

// CUDA
class cu_context: public context {
public:
  class context_switcher{
  public:
      context_switcher(driver::context const & ctx);
      ~context_switcher();
  private:
      driver::cu_context const & ctx_;
  };

private:
  static CUdevice get_device_of(CUcontext);

public:
  //Constructors
  cu_context(CUcontext cu, bool take_ownership = true);
  cu_context(driver::device* dev);
};

// OpenCL
class ocl_context: public context {
public:
  ocl_context(driver::device* dev);
};




}
}

#endif
