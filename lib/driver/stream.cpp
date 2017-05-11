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
#include <cassert>
#include <array>

#include "isaac/driver/backend.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/event.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/buffer.h"

namespace isaac
{

namespace driver
{

Stream::Stream(CUstream stream, bool take_ownership): cu_(stream, take_ownership)
{}

Stream::Stream(Context const & context): context_(context), cu_(CUstream(), true)
{
  ContextSwitcher ctx_switch(context_);
  dispatch::cuStreamCreate(&*cu_, 0);
}

void Stream::synchronize()
{
  ContextSwitcher ctx_switch(context_);
  dispatch::cuStreamSynchronize(*cu_);
}

Context const & Stream::context() const
{ return context_; }

void Stream::enqueue(Kernel const & kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<Event> const *, Event* event){
  ContextSwitcher ctx_switch(context_);
  if(event)
    dispatch::cuEventRecord(((cu_event_t)*event).first, *cu_);
  dispatch::cuLaunchKernel(kernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, *cu_,(void**)kernel.cu_params(), NULL);
  if(event)
    dispatch::cuEventRecord(((cu_event_t)*event).second, *cu_);
}

void Stream::write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr){
  if(blocking)
    dispatch::cuMemcpyHtoD(buffer + offset, ptr, size);
  else
    dispatch::cuMemcpyHtoDAsync(buffer + offset, ptr, size, *cu_);
}

void Stream::read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr){
  if(blocking)
    dispatch::cuMemcpyDtoH(ptr, buffer + offset, size);
  else
    dispatch::cuMemcpyDtoHAsync(ptr, buffer + offset, size, *cu_);
}

Handle<CUstream> const & Stream::cu() const
{ return cu_; }

}

}
