#include "isaac/driver/handle.h"
#include <memory>

namespace isaac
{

namespace driver
{

//CUDA
template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUcontext x) { cuda::check(dispatch::cuCtxDestroy(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUdeviceptr x) { cuda::check(dispatch::dispatch::cuMemFree(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUstream x) { cuda::check(dispatch::cuStreamDestroy(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUdevice) { }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUevent x) { cuda::check(dispatch::dispatch::cuEventDestroy(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUfunction) { }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUmodule x) { cuda::check(dispatch::dispatch::cuModuleUnload(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(cu_event_t x) { _delete(x.first); _delete(x.second); }

//OpenCL
template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_context x) { ocl::check(dispatch::clReleaseContext(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_mem x) { ocl::check(dispatch::clReleaseMemObject(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_command_queue x) { ocl::check(dispatch::clReleaseCommandQueue(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_device_id x) { ocl::check(dispatch::clReleaseDevice(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_event x) { ocl::check(dispatch::clReleaseEvent(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_kernel x) { ocl::check(dispatch::clReleaseKernel(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_program x) { ocl::check(dispatch::clReleaseProgram(x)); }

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
bool Handle<CLType, CUType>::operator==(Handle const & other) const
{
  if(backend_==CUDA && other.backend_==CUDA)
    return cu()==other.cu();
  if(backend_==OPENCL && other.backend_==OPENCL)
    return cl()==other.cl();
  return false;
}

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
{ return *cl_; }

template<class CLType, class CUType>
CLType const &  Handle<CLType, CUType>::cl() const
{ return *cl_; }

template<class CLType, class CUType>
CUType &  Handle<CLType, CUType>::cu()
{
    return *cu_;
}

template<class CLType, class CUType>
CUType const &  Handle<CLType, CUType>::cu() const
{
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
