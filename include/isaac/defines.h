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

#endif
