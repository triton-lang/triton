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

#ifdef __HIP_PLATFORM_AMD__
#include "triton/driver/handle_hip.h"
#else
#include "triton/driver/handle.h"
#endif
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

#ifdef __HIP_PLATFORM_AMD__
//HIP
inline void _delete(hipCtx_t x) { dispatch::hipCtxDestroy(x); }
inline void _delete(hipDeviceptr_t x) { dispatch::hipFree(x); }
inline void _delete(hipStream_t x) { dispatch::hipStreamDestroy(x); }
inline void _delete(hipDevice_t) { }
inline void _delete(hipEvent_t x) { dispatch::hipEventDestroy(x); }
#else
//CUDA
inline void _delete(CUcontext x) { dispatch::cuCtxDestroy(x); }
inline void _delete(CUdeviceptr x) { dispatch::cuMemFree(x); }
inline void _delete(CUstream x) { dispatch::cuStreamDestroy(x); }
inline void _delete(CUdevice) { }
inline void _delete(CUevent x) { dispatch::cuEventDestroy(x); }
#endif
inline void _delete(CUfunction) { }
inline void _delete(CUmodule x) { dispatch::cuModuleUnload(x); }
inline void _delete(cu_event_t x) { _delete(x.first); _delete(x.second); }
inline void _delete(CUPlatform){}

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


}
}
