#include "isaac/driver/handle.h"
#include <memory>

namespace isaac
{

namespace driver
{

#ifdef ISAAC_WITH_CUDA

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUcontext x) { cuCtxDestroy(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUdeviceptr x) { cuMemFree(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUstream x) { cuStreamDestroy(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUdevice) { }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUevent x) { cuEventDestroy(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUfunction) { }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(CUmodule x) { cuModuleUnload(x); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::_delete(std::pair<CUevent, CUevent> x) { _delete(x.first); _delete(x.second); }

#endif

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_context x) { ocl::check(clReleaseContext(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_mem x) { ocl::check(clReleaseMemObject(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_command_queue x) { ocl::check(clReleaseCommandQueue(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_device_id x) { ocl::check(clReleaseDevice(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_event x) { ocl::check(clReleaseEvent(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_kernel x) { ocl::check(clReleaseKernel(x)); }

template<class CLType, class CUType>
void Handle<CLType, CUType>::release(cl_program x) { ocl::check(clReleaseProgram(x)); }

template<class CLType, class CUType>
Handle<CLType, CUType>::Handle(backend_type backend, bool take_ownership): backend_(backend), has_ownership_(take_ownership)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA: cu_.reset(new CUType());
#endif
    case OPENCL: cl_.reset(new CLType());
  }
}

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator==(Handle const & other) const
{
#ifdef ISAAC_WITH_CUDA
  if(backend_==CUDA && other.backend_==CUDA)
    return cu()==other.cu();
#endif
  if(backend_==OPENCL && other.backend_==OPENCL)
    return cl()==other.cl();
  return false;
}

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator<(Handle const & other) const
{
#ifdef ISAAC_WITH_CUDA
  if(backend_==CUDA && other.backend_==CUDA)
    return (*cu_)<(*other.cu_);
#endif
  if(backend_==OPENCL && other.backend_==OPENCL)
    return (*cl_)<(*other.cl_);
#ifdef ISAAC_WITH_CUDA
  if(backend_==CUDA && other.backend_==OPENCL)
    return true;
#endif
  return false;
}

template<class CLType, class CUType>
Handle<CLType, CUType>::~Handle()
{
#ifdef ISAAC_WITH_CUDA
  if(has_ownership_ && cu_.unique())
    _delete(*cu_);
#endif
  if(has_ownership_ && cl_ && cl_.unique() && *cl_)
     release(*cl_);
}

template<class CLType, class CUType>
CLType &  Handle<CLType, CUType>::cl()
{ return *cl_; }

template<class CLType, class CUType>
CLType const &  Handle<CLType, CUType>::cl() const
{ return *cl_; }

#ifdef ISAAC_WITH_CUDA
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
template class Handle<cl_event, std::pair<CUevent, CUevent> >;
template class Handle<cl_kernel, CUfunction>;
template class Handle<cl_program, CUmodule>;
#else
template class Handle<cl_mem, void>;
template class Handle<cl_command_queue, void>;
template class Handle<cl_context, void>;
template class Handle<cl_device_id, void>;
template class Handle<cl_event, void>;
template class Handle<cl_kernel, void>;
template class Handle<cl_program, void>;
#endif


}
}
