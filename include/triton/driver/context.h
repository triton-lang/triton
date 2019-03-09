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

class context: public handle_interface<context, CUcontext>
{
private:
  static std::string get_cache_path();
  static CUdevice device(CUcontext);

public:
  //Constructors
  explicit context(CUcontext context, bool take_ownership = true);
  explicit context(driver::device const & dvc);
  //Accessors
  driver::device const & device() const;
  std::string const & cache_path() const;
  handle<CUcontext> const & cu() const;

private:
  handle<CUcontext> cu_;
  driver::device dvc_;
  std::string cache_path_;
};

class ContextSwitcher{
public:
    ContextSwitcher(driver::context const & ctx);
    ~ContextSwitcher();
private:
    driver::context const & ctx_;
};

}
}

#endif
