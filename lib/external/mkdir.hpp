#ifndef ISAAC_TOOLS_MKDIR
#define ISAAC_TOOLS_MKDIR

#include <cstring>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <errno.h>
#if defined(_WIN32)
  #include <direct.h>
#endif
namespace isaac
{

namespace tools
{

    inline int mkdir(std::string const & path)
    {
        #if defined(_WIN32)
            return _mkdir(path.c_str());
        #else
            return ::mkdir(path.c_str(), 0777);
        #endif
    }

    inline int mkpath(std::string const & path)
    {
        int status = 0;
        size_t pp = 0;
        size_t sp;
        while ((sp = path.find('/', pp)) != std::string::npos)
        {
            if (sp != pp){
                status = mkdir(path.substr(0, sp));
            }
            pp = sp + 1;
        }
        return (status==0 || errno==EEXIST)?0:-1;
    }

}

}

#endif
