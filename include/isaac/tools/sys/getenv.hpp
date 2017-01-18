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

#ifndef ISAAC_TOOLS_GETENV
#define ISAAC_TOOLS_GETENV

#include <string>
#include <cstdlib>

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
