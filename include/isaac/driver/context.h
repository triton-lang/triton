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

#ifndef ISAAC_DRIVER_CONTEXT_H
#define ISAAC_DRIVER_CONTEXT_H

#include <map>
#include <memory>
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class ISAACAPI Context: public has_handle_comparators<Context>
{
  friend class Program;
  friend class CommandQueue;
  friend class Buffer;

public:
  typedef Handle<cl_context, CUcontext> handle_type;

private:
  static std::string cache_path();

  static CUdevice device(CUcontext)
  {
      CUdevice res;
      dispatch::cuCtxGetDevice(&res);
      return res;
  }

public:
  //Constructors
  explicit Context(CUcontext const & context, bool take_ownership = true);
  explicit Context(cl_context const & context, bool take_ownership = true);
  explicit Context(Device const & device);
  //Accessors
  backend_type backend() const;
  Device const & device() const;
  handle_type const & handle() const;

private:
DISABLE_MSVC_WARNING_C4251
  backend_type backend_;
  Device device_;
  std::string cache_path_;
  handle_type h_;
RESTORE_MSVC_WARNING_C4251
};

}
}

#endif
