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

#ifndef TDL_INCLUDE_DRIVER_STREAM_H
#define TDL_INCLUDE_DRIVER_STREAM_H

#include <map>
#include "triton/driver/context.h"
#include "triton/driver/device.h"
#include "triton/driver/handle.h"
#include "triton/driver/buffer.h"

namespace triton
{

namespace driver
{

class kernel;
class Event;
class Range;
class buffer;

// Command Queue
class stream: public handle_interface<stream, CUstream>
{
public:
  //Constructors
  stream(CUstream stream, bool take_ownership);
  stream(driver::context const & context);

  //Accessors
  handle<CUstream> const & cu() const;
  driver::context const & context() const;

  //Synchronize
  void synchronize();

  //Enqueue
  void enqueue(kernel const & kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<Event> const * = NULL, Event *event = NULL);

  // Write
  void write(driver::buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr);

  template<class T> void write(driver::buffer const & buffer, bool blocking, std::size_t offset, std::vector<T> const & x)
  { write(buffer, blocking, offset, x.size()*sizeof(T), x.data()); }

  // Read
  void read(driver::buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr);

  template<class T> void read(driver::buffer const & buffer, bool blocking, std::size_t offset, std::vector<T>& x)
  { read(buffer, blocking, offset, x.size()*sizeof(T), x.data()); }
private:
  driver::context context_;
  handle<CUstream> cu_;
};


}

}

#endif
