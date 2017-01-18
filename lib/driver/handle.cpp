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

#include <cassert>
#include <memory>

#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{



//CUDA
template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUcontext x) { check_destruction(dispatch::cuCtxDestroy(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUdeviceptr x) { check_destruction(dispatch::cuMemFree(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUstream x) { check_destruction(dispatch::cuStreamDestroy(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUdevice) { }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUevent x) { check_destruction(dispatch::cuEventDestroy(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUfunction) { }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUmodule x) { check_destruction(dispatch::cuModuleUnload(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(cu_event_t x) { _delete(x.first); _delete(x.second); }

//OpenCL
template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_context x) { dispatch::clReleaseContext(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_mem x) { dispatch::clReleaseMemObject(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_command_queue x) { dispatch::clReleaseCommandQueue(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_device_id /*x*/) { /*dispatch::clReleaseDevice(x);*/ }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_event x) { dispatch::clReleaseEvent(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_kernel x) { dispatch::clReleaseKernel(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_program x) { dispatch::clReleaseProgram(x); }

template<class CLType, class CUType>
Handle<CLType, CUType>::Handle(backend_type backend, bool take_ownership): backend_(backend), has_ownership_(take_ownership)
{
  switch(backend_)
  {
    case CUDA: cu_.reset(new CUType());
    case OPENCL: cl_.reset(new CLType());
  }
}

template<class CLType, class CUType>
backend_type Handle<CLType, CUType>::backend() const
{ return backend_; }

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator==(Handle const & other) const
{
  if(backend_==CUDA && other.backend_==CUDA)
    return cu()==other.cu();
  if(backend_==OPENCL && other.backend_==OPENCL)
    return cl()==other.cl();
  return false;
}

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator!=(Handle const & other) const
{ return !((*this)==other); }

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator<(Handle const & other) const
{
  if(backend_==CUDA && other.backend_==CUDA)
    return (*cu_)<(*other.cu_);
  if(backend_==OPENCL && other.backend_==OPENCL)
    return (*cl_)<(*other.cl_);
  if(backend_==CUDA && other.backend_==OPENCL)
    return true;
  return false;
}

template<class CLType, class CUType>
Handle<CLType, CUType>::~Handle()
{
  if(backend_==CUDA && has_ownership_ && cu_ && cu_.unique() && *cu_){
    _delete(*cu_);
  }
  if(backend_==OPENCL && has_ownership_ && cl_ && cl_.unique() && *cl_)
     release(*cl_);
}

template<class CLType, class CUType>
CLType &  Handle<CLType, CUType>::cl()
{
    assert(backend_==OPENCL);
    return *cl_;
}

template<class CLType, class CUType>
CLType const &  Handle<CLType, CUType>::cl() const
{
    assert(backend_==OPENCL);
    return *cl_;
}

template<class CLType, class CUType>
CUType &  Handle<CLType, CUType>::cu()
{
    assert(backend_==CUDA);
    return *cu_;
}

template<class CLType, class CUType>
CUType const &  Handle<CLType, CUType>::cu() const
{
    assert(backend_==CUDA);
    return *cu_;
}

template class Handle<cl_mem, CUdeviceptr>;
template class Handle<cl_command_queue, CUstream>;
template class Handle<cl_context, CUcontext>;
template class Handle<cl_device_id, CUdevice>;
template class Handle<cl_event, cu_event_t>;
template class Handle<cl_kernel, CUfunction>;
template class Handle<cl_program, CUmodule>;

}
}
