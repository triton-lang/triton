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
Handle<CLType, CUType>::Handle(backend_type backend): backend_(backend)
{
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA: cu.reset(new CUType());
#endif
    case OPENCL: cl.reset(new CLType());
  }
}

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator==(Handle const & other) const
{
#ifdef ISAAC_WITH_CUDA
  if(backend_==CUDA && other.backend_==CUDA)
    return (*cu)==(*other.cu);
#endif
  if(backend_==OPENCL && other.backend_==OPENCL)
    return (*cl)()==(*other.cl)();
  return false;
}

template<class CLType, class CUType>
bool Handle<CLType, CUType>::operator<(Handle const & other) const
{
#ifdef ISAAC_WITH_CUDA
  if(backend_==CUDA && other.backend_==CUDA)
    return (*cu)<(*other.cu);
#endif
  if(backend_==OPENCL && other.backend_==OPENCL)
    return (*cl)()<(*other.cl)();
#ifdef ISAAC_WITH_CUDA
  if(backend_==CUDA && other.backend_==OPENCL)
    return true;
#endif
  return false;
}

template<class CLType, class CUType>
Handle<CLType, CUType>::~Handle()
{
  if(cu && cu.unique())
  {
    switch(backend_)
    {
#ifdef ISAAC_WITH_CUDA
      case CUDA: _delete(*cu); break;
#endif
      default: break;
    }
  }
}



#ifdef ISAAC_WITH_CUDA
template class Handle<cl::Buffer, CUdeviceptr>;
template class Handle<cl::CommandQueue, CUstream>;
template class Handle<cl::Context, CUcontext>;
template class Handle<cl::Device, CUdevice>;
template class Handle<cl::Event, std::pair<CUevent, CUevent> >;
template class Handle<cl::Kernel, CUfunction>;
template class Handle<cl::Program, CUmodule>;
#else
template class Handle<cl::Buffer, void>;
template class Handle<cl::CommandQueue, void>;
template class Handle<cl::Context, void>;
template class Handle<cl::Device, void>;
template class Handle<cl::Event, void>;
template class Handle<cl::Kernel, void>;
template class Handle<cl::Program, void>;
#endif


}
}
