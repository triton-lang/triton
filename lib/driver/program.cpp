#include <iostream>
#include <fstream>

#include "isaac/driver/program.h"
#include "isaac/driver/context.h"
#include "isaac/tools/sha1.hpp"

#ifdef ISAAC_WITH_CUDA
#include "helpers/cuda/vector.hpp"
#endif

namespace isaac
{

namespace driver
{

Program::Program(Context const & context, std::string const & source) : backend_(context.backend_), context_(context), source_(source), h_(backend_)
{
//  std::cout << source << std::endl;
  std::string cache_path = context.cache_path_;
  switch(backend_)
  {
#ifdef ISAAC_WITH_CUDA
    case CUDA:
    {

      std::string prefix = context_.device_.name() + "cuda";
      std::string sha1 = tools::sha1(prefix + source);
      std::string fname(cache_path + sha1);

      //Load cached program
      if(cache_path.size() && std::ifstream(fname, std::ios::binary))
      {
        cuda::check(cuModuleLoad(h_.cu.get(), fname.c_str()));
        break;
      }

      nvrtcProgram prog;

      const char * includes[] = {"helper_math.h"};
      const char * src[] = {helpers::cuda::vector};

      nvrtc::check(nvrtcCreateProgram(&prog, source.c_str(), NULL, 1, src, includes));
      try{
        const char * options[] = {"--gpu-architecture=compute_52", "--restrict"};
        nvrtc::check(nvrtcCompileProgram(prog, 2, options));
      }catch(nvrtc::exception::compilation const &)
      {
        size_t logsize;
        nvrtc::check(nvrtcGetProgramLogSize(prog, &logsize));
        std::string log(logsize, 0);
        nvrtc::check(nvrtcGetProgramLog(prog, (char*)log.data()));
        std::cout << "Compilation failed:" << std::endl;
        std::cout << log << std::endl;
      }

      size_t ptx_size;
      nvrtc::check(nvrtcGetPTXSize(prog, &ptx_size));
      std::vector<char> ptx(ptx_size);
      nvrtc::check(nvrtcGetPTX(prog, ptx.data()));
      cuda::check(cuModuleLoadDataEx(h_.cu.get(), ptx.data(), 0, NULL, NULL));

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
//    system(("perl /home/philippe/Development/maxas/maxas.pl -e " + sha1 + ".cubin > " + sha1 + ".sass").c_str());
//    system(("perl /home/philippe/Development/maxas/maxas.pl -i --noreuse" + sha1 + ".sass " + sha1 + ".cubin").c_str());

//    std::ifstream ifs(sha1 + ".cubin");
//    std::cout << sha1 << std::endl;
//    std::string str;

//    ifs.seekg(0, std::ios::end);
//    str.reserve(ifs.tellg());
//    ifs.seekg(0, std::ios::beg);

//    str.assign((std::istreambuf_iterator<char>(ifs)),
//                std::istreambuf_iterator<char>());
//    cuda::check(cuModuleLoadDataEx(h_.cu.get(), str.c_str(), 0, NULL, NULL));

      break;
    }
#endif
    case OPENCL:
    {
      std::vector<cl::Device> devices = context_.h_.cl().getInfo<CL_CONTEXT_DEVICES>();

      std::string prefix;
      for(std::vector<cl::Device >::const_iterator it = devices.begin(); it != devices.end(); ++it)
        prefix += it->getInfo<CL_DEVICE_NAME>() + it->getInfo<CL_DEVICE_VENDOR>() + it->getInfo<CL_DEVICE_VERSION>();
      std::string sha1 = tools::sha1(prefix + source);
      std::string fname(cache_path + sha1);

      //Load cached program
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
          h_.cl() = cl::Program(context_.h_.cl(), devices, cl::Program::Binaries(1, std::make_pair(cbuffer, len)));
          h_.cl().build();
          return;
        }
      }

      h_.cl() = cl::Program(context_.h_.cl(), source);
      try{
        ocl::check(h_.cl().build(devices));
      }catch(ocl::exception::build_program_failure const & e){
            for(std::vector< cl::Device >::const_iterator it = devices.begin(); it != devices.end(); ++it)
              std::cout << "Device : " << it->getInfo<CL_DEVICE_NAME>()
                      << "Build Status = " << h_.cl().getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*it) << std::endl
                      << "Build Log = " << h_.cl().getBuildInfo<CL_PROGRAM_BUILD_LOG>(*it) << std::endl;
      }

      //Save cached program
      if (cache_path.size())
      {
        std::ofstream cached(fname.c_str(),std::ios::binary);
        std::vector<std::size_t> sizes = h_.cl().getInfo<CL_PROGRAM_BINARY_SIZES>();
        cached.write((char*)&sizes[0], sizeof(std::size_t));
        std::vector<char*> binaries = h_.cl().getInfo<CL_PROGRAM_BINARIES>();
        cached.write((char*)binaries[0], std::streamsize(sizes[0]));
      }
      break;
    }
    default:
      throw;
  }
}

Context const & Program::context() const
{ return context_; }

}

}

