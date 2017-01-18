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

#include <iostream>
#include <fstream>

#include "isaac/driver/program.h"
#include "isaac/driver/context.h"

#include "isaac/exception/driver.h"

#include "helpers/cuda/vector.hpp"
#include "helpers/ocl/infos.hpp"

#include "tinysha1/sha1.hpp"

#include "isaac/tools/cpp/string.hpp"

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
        dispatch::cuModuleLoad(&h_.cu(), fname.c_str());
        break;
      }

      nvrtcProgram prog;

      const char * includes[] = {"vector.h"};
      const char * src[] = {helpers::cuda::vector};

      dispatch::nvrtcCreateProgram(&prog, source.c_str(), NULL, 1, src, includes);
      try{
        std::pair<unsigned int, unsigned int> capability = context_.device().nv_compute_capability();
        std::string capability_opt = "--gpu-architecture=compute_";
        capability_opt += tools::to_string(capability.first) + tools::to_string(capability.second);
        const char * options[] = {capability_opt.c_str(), "--restrict"};
        dispatch::nvrtcCompileProgram(prog, 2, options);
      }catch(exception::nvrtc::compilation const &)
      {
        size_t logsize;
        dispatch::nvrtcGetProgramLogSize(prog, &logsize);
        std::string log(logsize, 0);
        dispatch::nvrtcGetProgramLog(prog, (char*)log.data());
        std::cout << "Compilation failed:" << std::endl;
        std::cout << log << std::endl;
      }

      size_t ptx_size;
      dispatch::nvrtcGetPTXSize(prog, &ptx_size);
      std::vector<char> ptx(ptx_size);
      dispatch::nvrtcGetPTX(prog, ptx.data());
      dispatch::cuModuleLoadDataEx(&h_.cu(), ptx.data(), 0, NULL, NULL);

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
//    dispatch::cuModuleLoadDataEx(&h_.cu(), str.c_str(), 0, NULL, NULL);

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
          check(err);
          dispatch::clBuildProgram(h_.cl(), static_cast<cl_uint>(devices.size()), devices.data(), build_opt.c_str(), NULL, NULL);
          return;
        }
      }

      std::size_t srclen = source.size();
      const char * csrc = source.c_str();
      h_.cl() = dispatch::clCreateProgramWithSource(context_.h_.cl(), 1, &csrc, &srclen, &err);
      try{
        dispatch::clBuildProgram(h_.cl(), static_cast<cl_uint>(devices.size()), devices.data(), build_opt.c_str(), NULL, NULL);
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
      }catch(exception::ocl::build_program_failure const &){
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

Program::handle_type const & Program::handle() const
{ return h_; }

Context const & Program::context() const
{ return context_; }


}

}

