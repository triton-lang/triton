/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <hip/hip_common.h>

struct ihipModuleSymbol_t;
using hipFunction_t = ihipModuleSymbol_t*;

namespace hip_impl {

// This section contains internal APIs that
// needs to be exported
#ifdef __GNUC__
#pragma GCC visibility push (default)
#endif

struct kernarg_impl;
class kernarg {
public:
    kernarg();
    kernarg(kernarg&&);
    ~kernarg();
    std::uint8_t* data();
    std::size_t   size();
    void reserve(std::size_t);
    void resize(std::size_t);
private:
    kernarg_impl* impl;
};

class kernargs_size_align;
class program_state_impl;
class program_state {
public:
    program_state();
    ~program_state();
    program_state(const program_state&) = delete;

    hipFunction_t kernel_descriptor(std::uintptr_t,
                                    hsa_agent_t);

    kernargs_size_align get_kernargs_size_align(std::uintptr_t);
    hsa_executable_t load_executable(const char*, const size_t,
                                     hsa_executable_t,
                                     hsa_agent_t);
    hsa_executable_t load_executable_no_copy(const char*, const size_t,
                                             hsa_executable_t,
                                             hsa_agent_t);

    void* global_addr_by_name(const char* name);

private:
    friend class agent_globals_impl;
    program_state_impl* impl;
};

class kernargs_size_align {
public:
    std::size_t size(std::size_t n) const;
    std::size_t alignment(std::size_t n) const;
    const void* getHandle() const {return handle;};
private:
    const void* handle;
    friend kernargs_size_align program_state::get_kernargs_size_align(std::uintptr_t);
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

inline
__attribute__((visibility("hidden")))
program_state& get_program_state() {
    static program_state ps;
    return ps;
}
}  // Namespace hip_impl.
