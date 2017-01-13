/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
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
    //case HALF_TYPE: setArg(index, scal.values().float16); break;
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

