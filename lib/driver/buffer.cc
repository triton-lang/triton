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

#include "triton/driver/stream.h"
#include "triton/driver/buffer.h"
#include "triton/driver/context.h"
#include "triton/driver/dispatch.h"


namespace triton
{

namespace driver
{


//

buffer::buffer(size_t size, hipDeviceptr_t hip, bool take_ownership)
  : polymorphic_resource(hip, take_ownership), size_(size) {}

buffer::buffer(size_t size, CUdeviceptr cu, bool take_ownership)
  : polymorphic_resource(cu, take_ownership), size_(size) { }

buffer::buffer(size_t size, host_buffer_t hst, bool take_ownership)
  : polymorphic_resource(hst, take_ownership), size_(size) { }

size_t buffer::size() {
  return size_;
}

uintptr_t buffer::addr_as_uintptr_t() {
  switch(backend_){
    case HIP : return (uintptr_t)*hip_;
    case CUDA: return (uintptr_t)*cu_;
    case Host: return (uintptr_t)hst_->data;
    default: return 0;
  }
}


buffer* buffer::create(driver::context* ctx, size_t size) {
  switch(ctx->backend()){
//  case HIP : return new hip_buffer(size);
  case CUDA: return new cu_buffer(size);
  case Host: return new host_buffer(size);
  default: throw std::runtime_error("unknown backend");
  }
}

//

host_buffer::host_buffer(size_t size)
  :  buffer(size, host_buffer_t(), true){
  hst_->data = new char[size];
}


//
template<class T> struct _fn {} ;

template<> struct _fn<hipDeviceptr_t>{
  static constexpr auto alloc = dispatch::hipMalloc;
  static constexpr auto memset_async = dispatch::hipMemsetD8Async;
  static hipDeviceptr_t& h(buffer* buf) { return *buf->hip(); };
};

template<> struct _fn<CUdeviceptr>{
  static constexpr auto alloc = dispatch::cuMemAlloc_v2;
  static constexpr auto memset_async = dispatch::cuMemsetD8Async;
  static CUdeviceptr& h(buffer* buf) { return *buf->cu(); };
};

template<class T>
cu_hip_buffer<T>::cu_hip_buffer(size_t size)
  : buffer(size, T(), true) {
  _fn<T>::alloc(&_fn<T>::h(this), size);
}

template<class T>
cu_hip_buffer<T>::cu_hip_buffer(size_t size, T handle, bool take_ownership)
  : buffer(size, handle, take_ownership){
}

template<class T>
void cu_hip_buffer<T>::set_zero(driver::stream* queue, size_t size){
  _fn<T>::memset_async(_fn<T>::h(this), 0, size, *queue->cu());
}

}

}
