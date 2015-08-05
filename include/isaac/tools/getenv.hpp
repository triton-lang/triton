#ifndef ISAAC_TOOLS_GETENV
#define ISAAC_TOOLS_GETENV

#include <string>

namespace isaac
{

namespace tools
{

    inline std::string getenv(const char * name)
    {
        #ifdef _MSC_VER
            char* cache_path = 0;
            std::size_t sz = 0;
            _dupenv_s(&cache_path, &sz, name);
        #else
            const char * cache_path = std::getenv(name);
        #endif
        if(!cache_path)
            return "";
        std::string result(cache_path);
        #ifdef _MSC_VER
            free(cache_path);
        #endif
        return result;
    }

}

}

#endif
