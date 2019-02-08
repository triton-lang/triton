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
#include "driver/stream.h"
#include "driver/buffer.h"
#include "driver/context.h"
#include "driver/dispatch.h"


namespace tdl
{

namespace driver
{

Buffer::Buffer(Context const & context, size_t size) : context_(context)
{
  ContextSwitcher ctx_switch(context_);
  dispatch::cuMemAlloc(&*cu_, size);
}

Buffer::Buffer(Context const & context, CUdeviceptr cu, bool take_ownership):
  context_(context), cu_(cu, take_ownership)
{ }

void Buffer::set_zero(Stream const & queue, size_t size)
{
  ContextSwitcher ctx_switch(context_);
  dispatch::cuMemsetD8Async(*cu_, 0, size, queue);
}

Handle<CUdeviceptr> const & Buffer::cu() const
{ return cu_; }

Handle<CUdeviceptr> & Buffer::cu()
{ return cu_; }

}

}
