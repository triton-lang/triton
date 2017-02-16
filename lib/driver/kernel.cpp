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

#include "isaac/driver/kernel.h"
#include "isaac/driver/buffer.h"
#include "isaac/value_scalar.h"
#include <iostream>
#include <cstring>

namespace isaac
{

namespace driver
{

Kernel::Kernel(Program const & program, const char * name) : backend_(program.backend_), address_bits_(program.context().device().address_bits()), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      cu_params_store_.reserve(64);
      cu_params_.reserve(64);
      dispatch::cuModuleGetFunction(&h_.cu(), program.h_.cu(), name);\
      break;
    case OPENCL:
      cl_int err;
      h_.cl() = dispatch::clCreateKernel(program.h_.cl(), name, &err);
      check(err);
      break;
    default:
      throw;
  }
}

void Kernel::setArg(unsigned int index, value_scalar const & scal)
{
  switch(scal.dtype())
  {
    //case BOOL_TYPE: setArg(index, scal.values().bool8); break;
    case CHAR_TYPE: setArg(index, scal.values().int8); break;
    case UCHAR_TYPE: setArg(index, scal.values().uint8); break;
    case SHORT_TYPE: setArg(index, scal.values().int16); break;
    case USHORT_TYPE: setArg(index, scal.values().uint16); break;
    case INT_TYPE: setArg(index, scal.values().int32); break;
    case UINT_TYPE: setArg(index, scal.values().uint32); break;
    case LONG_TYPE: setArg(index, scal.values().int64); break;
    case ULONG_TYPE: setArg(index, scal.values().uint64); break;
    case FLOAT_TYPE: setArg(index, scal.values().float32); break;
    case DOUBLE_TYPE: setArg(index, scal.values().float64); break;
    default: throw unknown_datatype(scal.dtype());
  }
}

void Kernel::setArg(unsigned int index, std::size_t size, void* ptr)
{
  switch(backend_)
  {
    case CUDA:
      if(index + 1> cu_params_store_.size())
      {
        cu_params_store_.resize(index+1);
        cu_params_.resize(index+1);
      }
      cu_params_store_[index].reset(malloc(size), free);
      memcpy(cu_params_store_[index].get(), ptr, size);
      cu_params_[index] = cu_params_store_[index].get();
      break;
    case OPENCL:
      dispatch::clSetKernelArg(h_.cl(), index, size, ptr);
      break;
    default:
      throw;
  }
}

void Kernel::setArg(unsigned int index, Buffer const & data)
{
  switch(backend_)
  {
    case CUDA:
    {
      setArg(index, sizeof(CUdeviceptr), (void*)&data.h_.cu()); break;
    }
    case OPENCL:
      dispatch::clSetKernelArg(h_.cl(), index, sizeof(cl_mem), (void*)&data.h_.cl());
      break;
    default: throw;
  }
}

void Kernel::setSizeArg(unsigned int index, size_t N)
{
  switch(backend_)
  {
    case CUDA:
    {
      int NN = static_cast<cl_int>(N);
      setArg(index, sizeof(int), &NN);
      break;
    }
    case OPENCL:
    {
      cl_int NN = static_cast<cl_int>(N);
      setArg(index, 4, &NN);
      break;
    }
    default: throw;
  }
}

Kernel::handle_type const & Kernel::handle() const
{ return h_; }

}

}

