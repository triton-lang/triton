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

#include "isaac/driver/event.h"
#include "helpers/ocl/infos.hpp"

namespace isaac
{

namespace driver
{

Event::Event(backend_type backend) : backend_(backend), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      dispatch::cuEventCreate(&h_.cu().first, CU_EVENT_DEFAULT);
      dispatch::cuEventCreate(&h_.cu().second, CU_EVENT_DEFAULT);
      break;
    case OPENCL:
      break;
    default:
      throw;
  }
}

Event::Event(cl_event const & event, bool take_ownership) : backend_(OPENCL), h_(backend_, take_ownership)
{
  h_.cl() = event;
}

long Event::elapsed_time() const
{
  switch(backend_)
  {
    case CUDA:
      float time;
      dispatch::cuEventElapsedTime(&time, h_.cu().first, h_.cu().second);
      return 1e6*time;
    case OPENCL:
      return static_cast<long>(ocl::info<CL_PROFILING_COMMAND_END>(h_.cl()) - ocl::info<CL_PROFILING_COMMAND_START>(h_.cl()));
    default:
      throw;
  }
}

Event::handle_type const & Event::handle() const
{ return h_; }

}

}
