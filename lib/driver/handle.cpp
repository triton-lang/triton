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
inline void _delete(CUcontext x) { check_destruction(dispatch::cuCtxDestroy(x)); }
inline void _delete(CUdeviceptr x) { check_destruction(dispatch::cuMemFree(x)); }
inline void _delete(CUstream x) { check_destruction(dispatch::cuStreamDestroy(x)); }
inline void _delete(CUdevice) { }
inline void _delete(CUevent x) { check_destruction(dispatch::cuEventDestroy(x)); }
inline void _delete(CUfunction) { }
inline void _delete(CUmodule x) { check_destruction(dispatch::cuModuleUnload(x)); }
inline void _delete(cu_event_t x) { _delete(x.first); _delete(x.second); }
inline void _delete(cu_platform){}

//Constructor
template<class CUType>
Handle<CUType>::Handle(CUType cu, bool take_ownership): cu_(new CUType(cu)), has_ownership_(take_ownership)
{ }

template<class CUType>
Handle<CUType>::Handle(bool take_ownership): has_ownership_(take_ownership)
{ cu_.reset(new CUType()); }

//Accessors
template<class CUType>
bool Handle<CUType>::operator==(Handle const & other) const
{ return *cu_==*other.cu_; }

template<class CUType>
bool Handle<CUType>::operator!=(Handle const & other) const
{ return !((*this)==other); }

template<class CUType>
bool Handle<CUType>::operator<(Handle const & other) const
{ return (*cu_)<(*other.cu_); }

template<class CUType>
Handle<CUType>::~Handle(){
  if(has_ownership_ && cu_ && cu_.unique() && *cu_)
    _delete(*cu_);
}

template<class CUType>
Handle<CUType>::operator CUType() const
{ return *cu_; }

template class Handle<CUdeviceptr>;
template class Handle<CUstream>;
template class Handle<CUcontext>;
template class Handle<CUdevice>;
template class Handle<cu_event_t>;
template class Handle<CUfunction>;
template class Handle<CUmodule>;
template class Handle<cu_platform>;

}
}
