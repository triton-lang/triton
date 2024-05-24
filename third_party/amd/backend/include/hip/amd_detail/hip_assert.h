/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

// abort
extern "C" __device__ inline __attribute__((weak))
void abort() {
  __builtin_trap();
}

// The noinline attribute helps encapsulate the printf expansion,
// which otherwise has a performance impact just by increasing the
// size of the calling function. Additionally, the weak attribute
// allows the function to exist as a global although its definition is
// included in every compilation unit.
#if defined(_WIN32) || defined(_WIN64)
extern "C" __device__ __attribute__((noinline)) __attribute__((weak))
void _wassert(const wchar_t *_msg, const wchar_t *_file, unsigned _line) {
    // FIXME: Need `wchar_t` support to generate assertion message.
    __builtin_trap();
}
#else /* defined(_WIN32) || defined(_WIN64) */
extern "C" __device__ __attribute__((noinline)) __attribute__((weak))
void __assert_fail(const char *assertion,
                   const char *file,
                   unsigned int line,
                   const char *function)
{
  const char fmt[] = "%s:%u: %s: Device-side assertion `%s' failed.\n";

  // strlen is not available as a built-in yet, so we create our own
  // loop in a macro. With a string literal argument, the compiler
  // usually manages to replace the loop with a constant.
  //
  // The macro does not check for null pointer, since all the string
  // arguments are defined to be constant literals when called from
  // the assert() macro.
  //
  // NOTE: The loop below includes the null terminator in the length
  // as required by append_string_n().
#define __hip_get_string_length(LEN, STR)       \
  do {                                          \
    const char *tmp = STR;                      \
    while (*tmp++);                             \
    LEN = tmp - STR;                            \
  } while (0)

  auto msg = __ockl_fprintf_stderr_begin();
  int len = 0;
  __hip_get_string_length(len, fmt);
  msg = __ockl_fprintf_append_string_n(msg, fmt, len, 0);
  __hip_get_string_length(len, file);
  msg = __ockl_fprintf_append_string_n(msg, file, len, 0);
  msg = __ockl_fprintf_append_args(msg, 1, line, 0, 0, 0, 0, 0, 0, 0);
  __hip_get_string_length(len, function);
  msg = __ockl_fprintf_append_string_n(msg, function, len, 0);
  __hip_get_string_length(len, assertion);
  __ockl_fprintf_append_string_n(msg, assertion, len, /* is_last = */ 1);

#undef __hip_get_string_length

  __builtin_trap();
}

extern "C" __device__ __attribute__((noinline)) __attribute__((weak))
void __assertfail()
{
    // ignore all the args for now.
    __builtin_trap();
}
#endif /* defined(_WIN32) || defined(_WIN64) */

#if defined(NDEBUG)
#define __hip_assert(COND)
#else
#define __hip_assert(COND)                          \
  do {                                              \
    if (!(COND))                                    \
      __builtin_trap();                             \
  } while (0)
#endif
