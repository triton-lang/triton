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
#include <iostream>
#include <fstream>

#include "isaac/driver/program.h"
#include "isaac/driver/context.h"

#include "helpers/cuda/vector.hpp"
#include "helpers/ocl/infos.hpp"

#include "tinysha1/sha1.hpp"

#include "cpp/to_string.hpp"

namespace isaac
{

namespace driver
{

Program::Program(Context const & context, std::string const & source) : backend_(context.backend_), context_(context), source_(source), h_(backend_, true)
{
//  std::cout << source << std::endl;
  std::string cache_path = context.cache_path_;
  switch(backend_)
  {
    case CUDA:
    {
      std::string prefix = context_.device_.name() + "cuda";
      std::string sha1 = tools::sha1(prefix + source);
      std::string fname(cache_path + sha1);

      //Load cached program
      if(cache_path.size() && std::ifstream(fname, std::ios::binary))
      {
        cuda::check(dispatch::cuModuleLoad(&h_.cu(), fname.c_str()));
        break;
      }

      nvrtcProgram prog;

      const char * includes[] = {"helper_math.h"};
      const char * src[] = {helpers::cuda::vector};

      nvrtc::check(dispatch::nvrtcCreateProgram(&prog, source.c_str(), NULL, 1, src, includes));
      try{
        std::pair<unsigned int, unsigned int> capability = context_.device().nv_compute_capability();
        std::string capability_opt = "--gpu-architecture=compute_";
        capability_opt += tools::to_string(capability.first) + tools::to_string(capability.second);
        const char * options[] = {capability_opt.c_str(), "--restrict"};
        nvrtc::check(dispatch::nvrtcCompileProgram(prog, 2, options));
      }catch(nvrtc::exception::compilation const &)
      {
        size_t logsize;
        nvrtc::check(dispatch::nvrtcGetProgramLogSize(prog, &logsize));
        std::string log(logsize, 0);
        nvrtc::check(dispatch::nvrtcGetProgramLog(prog, (char*)log.data()));
        std::cout << "Compilation failed:" << std::endl;
        std::cout << log << std::endl;
      }

      size_t ptx_size;
      nvrtc::check(dispatch::nvrtcGetPTXSize(prog, &ptx_size));
      std::vector<char> ptx(ptx_size);
      nvrtc::check(dispatch::nvrtcGetPTX(prog, ptx.data()));
      cuda::check(dispatch::cuModuleLoadDataEx(&h_.cu(), ptx.data(), 0, NULL, NULL));

      //Save cached program
      if (cache_path.size())
      {
        std::ofstream cached(fname.c_str(),std::ios::binary);
        cached.write((char*)ptx.data(), std::streamsize(ptx_size));
      }

//    std::ofstream oss(sha1 + ".cu", std::ofstream::out | std::ofstream::trunc);
//    oss << source << std::endl;
//    oss.close();

//    system(("/usr/local/cuda-7.0/bin/nvcc " + sha1 + ".cu -gencode arch=compute_50,code=sm_50 -cubin").c_str());
//    system(("perl /maxas.pl -e " + sha1 + ".cubin > " + sha1 + ".sass").c_str());
//    system(("perl /maxas.pl -i --noreuse" + sha1 + ".sass " + sha1 + ".cubin").c_str());

//    std::ifstream ifs(sha1 + ".cubin");
//    std::cout << sha1 << std::endl;
//    std::string str;

//    ifs.seekg(0, std::ios::end);
//    str.reserve(ifs.tellg());
//    ifs.seekg(0, std::ios::beg);

//    str.assign((std::istreambuf_iterator<char>(ifs)),
//                std::istreambuf_iterator<char>());
//    cuda::check(dispatch::cuModuleLoadDataEx(&h_.cu(), str.c_str(), 0, NULL, NULL));

      break;
    }
    case OPENCL:
    {
      cl_int err;
      std::vector<cl_device_id> devices = ocl::info<CL_CONTEXT_DEVICES>(context_.h_.cl());

      std::string prefix;
      for(cl_device_id dev: devices)
        prefix += ocl::info<CL_DEVICE_NAME>(dev) + ocl::info<CL_DEVICE_VENDOR>(dev) + ocl::info<CL_DEVICE_VERSION>(dev);
      std::string sha1 = tools::sha1(prefix + source);
      std::string fname(cache_path + sha1);
      //Load cached program
      std::string build_opt;
      if(cache_path.size())
      {
        std::ifstream cached(fname, std::ios::binary);
        if (cached)
        {
          std::size_t len;
          std::vector<char> buffer;
          cached.read((char*)&len, sizeof(std::size_t));
          buffer.resize(len);
          cached.read((char*)buffer.data(), std::streamsize(len));
          char* cbuffer = buffer.data();
          h_.cl() = dispatch::clCreateProgramWithBinary(context_.h_.cl(), static_cast<cl_uint>(devices.size()), devices.data(), &len, (const unsigned char **)&cbuffer, NULL, &err);
          ocl::check(err);
          ocl::check(dispatch::clBuildProgram(h_.cl(), static_cast<cl_uint>(devices.size()), devices.data(), build_opt.c_str(), NULL, NULL));
          return;
        }
      }

      std::size_t srclen = source.size();
      const char * csrc = source.c_str();
      h_.cl() = dispatch::clCreateProgramWithSource(context_.h_.cl(), 1, &csrc, &srclen, &err);
      try{
        ocl::check(dispatch::clBuildProgram(h_.cl(), static_cast<cl_uint>(devices.size()), devices.data(), build_opt.c_str(), NULL, NULL));
        //Save cached program
        if (cache_path.size())
        {
          std::ofstream cached(fname.c_str(),std::ios::binary);
          std::vector<std::size_t> sizes = ocl::info<CL_PROGRAM_BINARY_SIZES>(h_.cl());
          cached.write((char*)&sizes[0], sizeof(std::size_t));
          std::vector<unsigned char*> binaries = ocl::info<CL_PROGRAM_BINARIES>(h_.cl());
          cached.write((char*)binaries[0], std::streamsize(sizes[0]));
          for(unsigned char * ptr: binaries)
              delete[] ptr;
        }
      }catch(ocl::exception::build_program_failure const &){
            for(std::vector<cl_device_id>::const_iterator it = devices.begin(); it != devices.end(); ++it)
            {
              std::cout << "Device : " << ocl::info<CL_DEVICE_NAME>(*it)
                        << "Build Status = " << ocl::info<CL_PROGRAM_BUILD_STATUS>(h_.cl(), *it) << std::endl
                        << "Build Log = " << ocl::info<CL_PROGRAM_BUILD_LOG>(h_.cl(),*it) << std::endl;
            }
      }


      break;
    }
    default:
      throw;
  }
}

Context const & Program::context() const
{
    return context_;
}


}

}

