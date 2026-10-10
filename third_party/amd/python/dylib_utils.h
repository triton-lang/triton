#ifndef TRITON_THIRD_PARTY_AMD_PYTHON_DYLIB_UTILS_H
#define TRITON_THIRD_PARTY_AMD_PYTHON_DYLIB_UTILS_H

#include <nanobind/nanobind.h>
#include <optional>
#include <string>

// Returns the path of the first loaded library whose basename contains
// |libraryName|, or nullopt if none matches. Does not touch the GIL: python
// callers holding it must release it first, because dl_iterate_phdr acquires
// the dynamic linker lock and holding the GIL while blocking on that lock
// deadlocks against threads acquiring the two locks in the opposite order.
std::optional<std::string> findLoadedLibrary(const std::string &libraryName);

void init_triton_amd_loader(nanobind::module_ &m);

#endif // TRITON_THIRD_PARTY_AMD_PYTHON_DYLIB_UTILS_H
