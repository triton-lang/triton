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

#include "triton/driver/handle.h"
#include "triton/driver/error.h"

namespace triton
{

namespace driver
{

//Host
inline void _delete(host_platform_t) { }
inline void _delete(host_device_t)   { }
inline void _delete(host_context_t)  { }
inline void _delete(host_module_t)   { }
inline void _delete(host_stream_t)   { }
inline void _delete(host_buffer_t x)   { if(x.data) delete[] x.data; }
inline void _delete(host_function_t) { }

//CUDA
inline CUresult _delete(CUcontext x)   { return dispatch::cuCtxDestroy(x); }
inline CUresult _delete(CUdeviceptr x) { return dispatch::cuMemFree(x); }
inline CUresult _delete(CUstream x)    { return dispatch::cuStreamDestroy(x); }
inline CUresult _delete(CUdevice)      { return CUDA_SUCCESS; }
inline CUresult _delete(CUevent x)     { return dispatch::cuEventDestroy(x); }
inline CUresult _delete(CUfunction)    { return CUDA_SUCCESS; }
inline CUresult _delete(CUmodule x)    { return dispatch::cuModuleUnload(x); }
inline CUresult _delete(cu_event_t x)  { _delete(x.first); return _delete(x.second); }
inline CUresult _delete(CUPlatform)    { return CUDA_SUCCESS; }

//HIP
inline hipError_t _delete(hipCtx_t x)       { return dispatch::hipCtxDestroy(x); }
inline hipError_t _delete(hipDeviceptr_t x) { return dispatch::hipFree(x); }
inline hipError_t _delete(hipStream_t x)    { return dispatch::hipStreamDestroy(x); }
// hipDevice maps to `int` like CUdevice. No redefinition here
inline hipError_t _delete(hipEvent_t x)     { return dispatch::hipEventDestroy(x); }
inline hipError_t _delete(hipFunction_t)    { return hipSuccess;}
inline hipError_t _delete(hipModule_t x)    { return dispatch::hipModuleUnload(x); }
inline hipError_t _delete(hipPlatform)      { return hipSuccess;  }
inline hipError_t _delete(hip_event_t x)    { _delete(x.first); return _delete(x.second); }

//Constructor

template<class T>
handle<T>::handle(T cu, bool take_ownership): h_(new T(cu)), has_ownership_(take_ownership)
{ }

template<class T>
handle<T>::handle(): has_ownership_(false){ }


template<class T>
handle<T>::~handle(){
  try{
    if(has_ownership_ && h_ && h_.unique())
      _delete(*h_);
  }catch(const exception::cuda::base&){
    // order of destruction for global variables
    // is not guaranteed
  }
}

template class handle<CUdeviceptr>;
template class handle<CUstream>;
template class handle<CUcontext>;
template class handle<CUdevice>;
template class handle<cu_event_t>;
template class handle<CUfunction>;
template class handle<CUmodule>;
template class handle<CUPlatform>;

template class handle<host_platform_t>;
template class handle<host_device_t>;
template class handle<host_context_t>;
template class handle<host_module_t>;
template class handle<host_stream_t>;
template class handle<host_buffer_t>;
template class handle<host_function_t>;

template class handle<hipDeviceptr_t>;
template class handle<hipStream_t>;
template class handle<hipCtx_t>;
//template class handle<hipDevice_t>;
template class handle<hip_event_t>;
template class handle<hipFunction_t>;
template class handle<hipModule_t>;
template class handle<hipPlatform>;
}
}
