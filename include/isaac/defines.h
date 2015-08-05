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
