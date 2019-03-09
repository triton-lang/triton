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

#ifndef TDL_INCLUDE_DRIVER_BUFFER_H
#define TDL_INCLUDE_DRIVER_BUFFER_H

#include "triton/driver/handle.h"
#include "triton/driver/context.h"

namespace triton
{
namespace driver
{

class stream;

// Buffer
class buffer: public handle_interface<buffer, CUdeviceptr>
{
public:
  buffer(driver::context const & context, size_t size);
  buffer(driver::context const & context, CUdeviceptr cu, bool take_ownership);
  void set_zero(stream const & queue, size_t size);
  handle<CUdeviceptr> const & cu() const;
  handle<CUdeviceptr> & cu();

private:
  context context_;
  handle<CUdeviceptr> cu_;
};

}
}

#endif
