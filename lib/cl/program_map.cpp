#include <map>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "atidlas/cl/program_map.h"
#include "sha1.hpp"

namespace atidlas
{

namespace cl_ext
{

program_map::program_map()
{
  if (std::getenv("ATIDLAS_CACHE_PATH"))
    cache_path_ = std::getenv("ATIDLAS_CACHE_PATH");
  else
    cache_path_ = "";
}

cl::Program program_map::add(cl::Context & context, std::string const & pname, std::string const & source)
{

  cl_context clctx = context.operator ()();
  container_type::mapped_type map = data_[clctx];
  container_type::mapped_type::iterator it = map.find(pname);
  if(it!=map.end())
    return it->second;

  cl_int err;
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

  cl::Program res;

  // Retrieves the program in the cache
  bool compile = true;
  if (cache_path_.size())
  {
    std::string prefix;
    for(std::vector< cl::Device >::const_iterator it = devices.begin(); it != devices.end(); ++it)
      prefix += it->getInfo<CL_DEVICE_NAME>() + it->getInfo<CL_DEVICE_VENDOR>() + it->getInfo<CL_DEVICE_VERSION>();
    std::string sha1 = tools::sha1(prefix + source);

    std::ifstream cached((cache_path_+sha1).c_str(),std::ios::binary);
    if (cached)
    {
      std::size_t len;
      std::vector<char> buffer;
      cached.read((char*)&len, sizeof(std::size_t));
      buffer.resize(len);
      cached.read((char*)buffer.data(), std::streamsize(len));
      char* cbuffer = buffer.data();
      res = cl::Program(context, devices, cl::Program::Binaries(1, std::make_pair(cbuffer, len)), NULL, &err);
      compile = false;
    }
  }
  //Gets from source
  if(compile)
  {
    const char * csrc = source.c_str();
    std::size_t srclen = source.size();
    res = cl::Program(context, cl::Program::Sources(1, std::make_pair(csrc, srclen)));
  }

  try{
    err = res.build(devices);
  }catch(cl::Error const & e){
      if (err != CL_SUCCESS)
        for(std::vector< cl::Device >::const_iterator it = devices.begin(); it != devices.end(); ++it)
          std::cout << "Device : " << it->getInfo<CL_DEVICE_NAME>()
                  << "Build Status = " << res.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*it) << std::endl
                  << "Build Log = " << res.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*it) << std::endl;
  }


  // Store the program in the cache
  if (cache_path_.size() && compile)
  {
    std::vector<std::size_t> sizes = res.getInfo<CL_PROGRAM_BINARY_SIZES>();
    std::vector<char*> binaries = res.getInfo<CL_PROGRAM_BINARIES>();

    std::string prefix;
    for(std::vector< cl::Device >::const_iterator it = devices.begin(); it != devices.end(); ++it)
      prefix += it->getInfo<CL_DEVICE_NAME>() + it->getInfo<CL_DEVICE_VENDOR>() + it->getInfo<CL_DEVICE_VERSION>();
    std::string sha1 = tools::sha1(prefix + source);
    std::ofstream cached((cache_path_+sha1).c_str(),std::ios::binary);

    cached.write((char*)&sizes[0], sizeof(std::size_t));
    cached.write((char*)binaries[0], std::streamsize(sizes[0]));
  }

  map[pname] = res;
  return map[pname];
}

cl::Program & program_map::at(cl::Context const & context, std::string const & key)
{
  return data_[context()].at(key);
}

void program_map::erase(cl::Context const & context, std::string const & pname)
{
  container_type::mapped_type & map = data_[context()];
  container_type::mapped_type::iterator it = map.find(pname);
  if(it!=map.end())
    map.erase(it);
}

program_map pmap = program_map();

}

}
