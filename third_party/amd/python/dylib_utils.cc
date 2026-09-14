#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "dylib_utils.h"
#include <cstddef>
#include <cstring>
#include <nanobind/nanobind.h>
#include <optional>
#include <stdexcept>

#if defined(__linux__)
#include <link.h>
#endif

namespace py = nanobind;

namespace {

#if defined(__linux__)
constexpr size_t maxPathLength = 4096;

struct FindLoadedLibraryData {
  const char *libraryName;
  char path[maxPathLength + 1];
  bool found;
  bool pathTooLong;
};

// dl_iterate_phdr invokes this callback while holding the dynamic linker lock.
// Do not call Python, allocate memory, or throw from this function.
int findLoadedLibraryCallback(dl_phdr_info *info, size_t,
                              void *opaque) noexcept {
  auto *data = static_cast<FindLoadedLibraryData *>(opaque);
  const char *path = info->dlpi_name;
  if (path == nullptr || path[0] == '\0')
    return 0;

  const char *basename = std::strrchr(path, '/');
  basename = basename == nullptr ? path : basename + 1;
  if (std::strstr(basename, data->libraryName) == nullptr)
    return 0;

  size_t pathLength = ::strnlen(path, maxPathLength + 1);
  if (pathLength > maxPathLength) {
    data->pathTooLong = true;
    return 1;
  }

  std::memcpy(data->path, path, pathLength + 1);
  data->found = true;
  return 1;
}
#endif

} // namespace

std::optional<std::string> findLoadedLibrary(const std::string &libraryName) {
#if defined(__linux__)
  FindLoadedLibraryData data{};
  data.libraryName = libraryName.c_str();
  dl_iterate_phdr(findLoadedLibraryCallback, &data);
  if (data.pathTooLong)
    throw std::runtime_error("loaded library path exceeds 4096 bytes");
  if (!data.found)
    return std::nullopt;
  return std::string(data.path);
#else
  return std::nullopt;
#endif
}

void init_triton_amd_loader(py::module_ &m) {
  m.def(
      "find_loaded_library",
      [](const char *libraryName) -> py::object {
        std::optional<std::string> result;
        {
          py::gil_scoped_release release;
          result = findLoadedLibrary(libraryName);
        }
        if (!result)
          return py::none();
        PyObject *path = PyUnicode_DecodeFSDefault(result->c_str());
        if (path == nullptr)
          throw py::python_error();
        return py::steal<py::object>(path);
      },
      py::arg("library_name"),
      "Return the path of the first loaded library matching the name.");
}
