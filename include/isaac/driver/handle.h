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

#ifndef ISAAC_DRIVER_HANDLE_H
#define ISAAC_DRIVER_HANDLE_H

#include <memory>

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include <iostream>
namespace isaac
{

namespace driver
{

struct cu_event_t{
    operator bool() const { return first && second; }
    CUevent first;
    CUevent second;
};

template<class CLType, class CUType>
class ISAACAPI Handle
{
private:
  static void _delete(CUcontext x);
  static void _delete(CUdeviceptr x);
  static void _delete(CUstream x);
  static void _delete(CUdevice);
  static void _delete(CUevent x);
  static void _delete(CUfunction);
  static void _delete(CUmodule x);
  static void _delete(cu_event_t x);

  static void release(cl_context x);
  static void release(cl_mem x);
  static void release(cl_command_queue x);
  static void release(cl_device_id x);
  static void release(cl_event x);
  static void release(cl_kernel x);
  static void release(cl_program x);

public:
  //Constructors
  Handle(backend_type backend, bool take_ownership = true);
  //Comparison
  bool operator==(Handle const & other) const;
  bool operator!=(Handle const & other) const;
  bool operator<(Handle const & other) const;
  //Accessors
  backend_type backend() const;
  CLType & cl();
  CLType const & cl() const;
  CUType & cu();
  CUType const & cu() const;
  ~Handle();

private:
DISABLE_MSVC_WARNING_C4251
  std::shared_ptr<CLType> cl_;
  std::shared_ptr<CUType> cu_;
RESTORE_MSVC_WARNING_C4251

private:
  backend_type backend_;
  bool has_ownership_;
};

//Helper for automatic implementation of comparison operators
template<class T>
class has_handle_comparators
{
public:
  friend bool operator==(T const & x, T const & y) { return x.handle() == y.handle(); }
  friend bool operator!=(T const & x, T const & y) { return x.handle() != y.handle(); }
  friend bool operator<(T const & x, T const & y) { return x.handle() < y.handle(); }
};

}
}

#endif
