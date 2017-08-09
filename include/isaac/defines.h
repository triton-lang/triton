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

#ifndef ISAAC_DEFINES_H
#define ISAAC_DEFINES_H

#if defined(_WIN32) || defined(_MSC_VER)
    #ifdef ISAAC_DLL
        #define ISAACAPI  __declspec(dllexport)
    #else
        #define ISAACAPI  __declspec(dllimport)
    #endif
#else
    #define ISAACAPI   __attribute__((visibility("default")))
#endif

//In Windows, the dllexport and dllimport attributes have to be declared manually,
//although this is unnecessary in Linux. therefore the ISAACWINAPI is declared
//here to process this situation separately in Windows and Linux.
#if defined(_WIN32) || defined(_MSC_VER)
    #define ISAACWINAPI ISAACAPI
#else
    #define ISAACWINAPI
#endif

//In Windows, the dllexport and dllimport attributes cannot be used with
//template together, therefore the ISAACNOTWINAPI is declared here to process
//this situation separately in Windows and Linux.
#if defined(_WIN32) || defined(_MSC_VER)
#define ISAACNOTWINAPI
#else
#define ISAACNOTWINAPI ISAACAPI
#endif

#if defined(_WIN32) || defined(_MSC_VER)
	#define DISABLE_MSVC_WARNING_C4251 __pragma(warning(disable: 4251))
	#define RESTORE_MSVC_WARNING_C4251 __pragma(warning(default: 4251))
	#define DISABLE_MSVC_WARNING_C4275 __pragma(warning(disable: 4275))
	#define RESTORE_MSVC_WARNING_C4275 __pragma(warning(disable: 4275))

#else
    #define DISABLE_MSVC_WARNING_C4251
    #define RESTORE_MSVC_WARNING_C4251
    #define DISABLE_MSVC_WARNING_C4275
    #define RESTORE_MSVC_WARNING_C4275
#endif

#endif
