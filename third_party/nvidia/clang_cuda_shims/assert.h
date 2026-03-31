#pragma once

#if defined(__APPLE__) && defined(__CUDA__)

#undef assert

#ifdef NDEBUG
#define assert(e) ((void)0)
#else
#ifdef __FILE_NAME__
#define __ASSERT_FILE_NAME __FILE_NAME__
#else
#define __ASSERT_FILE_NAME __FILE__
#endif

#ifdef __cplusplus
#define assert(e)                                                              \
  (static_cast<void>(__builtin_expect(!!(e), 1)                                \
                         ? 0                                                   \
                         : (__assert_fail(#e, __ASSERT_FILE_NAME, __LINE__,    \
                                          __PRETTY_FUNCTION__),                \
                            0)))
#else
#define assert(e)                                                              \
  ((void)(__builtin_expect(!!(e), 1)                                           \
              ? 0                                                              \
              : (__assert_fail(#e, __ASSERT_FILE_NAME, __LINE__, __func__),    \
                 0)))
#endif

#endif

#include <_static_assert.h>

#else

#include_next <assert.h>

#endif
