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

#ifndef ISAAC_DRIVER_STREAM_H
#define ISAAC_DRIVER_STREAM_H

#include <map>
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class Kernel;
class Event;
class Range;
class Buffer;

// Command Queue
class Stream: public  Handle<CUstream>
{
  typedef Handle<CUstream> base_type;

public:
  //Constructors
  using base_type::base_type;
  Stream(Context const & context);
  //Accessors
  Context const & context() const;
  //Synchronize
  void synchronize();
  //Enqueue calls
  void enqueue(Kernel const & kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<Event> const * = NULL, Event *event = NULL);
  void write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr);

private:
  Context context_;
};


}

}

#endif
