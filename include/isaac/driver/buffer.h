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

#ifndef ISAAC_DRIVER_BUFFER_H
#define ISAAC_DRIVER_BUFFER_H

#include "isaac/types.h"
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/handle.h"
#include "isaac/driver/dispatch.h"
namespace isaac
{

namespace driver
{

// Buffer
class ISAACAPI Buffer: public has_handle_comparators<Buffer>
{
public:
  typedef Handle<cl_mem, CUdeviceptr> handle_type;

private:
  friend class CommandQueue;
  friend class Kernel;
  //Wrapper to get CUDA context from Memory
  static CUcontext context(CUdeviceptr h)
  {
      CUcontext res;
      check(dispatch::cuPointerGetAttribute((void*)&res, CU_POINTER_ATTRIBUTE_CONTEXT, h));
      return res;
  }

public:
  //Constructors
  Buffer(CUdeviceptr h, bool take_ownership = true);
  Buffer(cl_mem Buffer, bool take_ownership = true);
  Buffer(Context const & context, size_t size);
  //Accessors
  handle_type&  handle();
  handle_type const &  handle() const;
  Context const & context() const;
private:
  backend_type backend_;
  Context context_;
  handle_type h_;
};

inline Buffer make_buffer(backend_type backend, cl_mem clh = 0, CUdeviceptr cuh = 0, bool take_ownership = true)
{
  if(backend==OPENCL)
    return Buffer(clh, take_ownership);
  else
    return Buffer(cuh, take_ownership);
}


}

}

#endif
