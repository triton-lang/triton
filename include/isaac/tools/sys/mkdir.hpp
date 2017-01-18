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
