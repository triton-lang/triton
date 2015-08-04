#include "isaac/driver/program_cache.h"

namespace isaac
{

namespace driver
{

Program & ProgramCache::add(Context const & context, std::string const & name, std::string const & src)
{
    std::map<std::string, Program>::iterator it = cache_.find(name);
    if(it==cache_.end())
    {
        std::string extensions;
        std::string ext = "cl_khr_fp64";
        if(context.device().extensions().find(ext)!=std::string::npos)
          extensions = "#pragma OPENCL EXTENSION " + ext + " : enable\n";
        return cache_.insert(std::make_pair(name, driver::Program(context, extensions + src))).first->second;
    }
    return it->second;
}

Program const * ProgramCache::find(const std::string &name)
{
    std::map<std::string, Program>::const_iterator it = cache_.find(name);
    if(it==cache_.end())
        return NULL;
    return &(it->second);
}

void ProgramCache::clear()
{
    cache_.clear();
}

}

}

