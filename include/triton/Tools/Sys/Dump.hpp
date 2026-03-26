#ifndef TRITON_TOOLS_SYS_DUMP_HPP
#define TRITON_TOOLS_SYS_DUMP_HPP

#include <memory>

#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::tools {

inline std::unique_ptr<llvm::raw_fd_ostream> createMlirDumpStream() {
  std::error_code EC;
  std::string dumpPath = getStrEnv("MLIR_DUMP_PATH");

  llvm::sys::fs::file_status status;
  bool pathIsDir = !llvm::sys::fs::status(dumpPath, status) &&
                   llvm::sys::fs::is_directory(status);
  if (pathIsDir) {
    llvm::SmallString<256> model(dumpPath);
    llvm::sys::path::append(model, "triton-%%%%%%.mlir");
    int fd = -1;
    llvm::SmallString<256> uniquePath;
    EC = llvm::sys::fs::createUniqueFile(model, fd, uniquePath);
    assert(!EC && "failed to create a dump file under MLIR_DUMP_PATH");
    return std::make_unique<llvm::raw_fd_ostream>(fd, /*shouldClose=*/true);
  }

  auto stream = std::make_unique<llvm::raw_fd_ostream>(
      dumpPath, EC, llvm::sys::fs::CD_CreateAlways);
  assert(!EC && "failed to open MLIR_DUMP_PATH");
  return stream;
}

inline llvm::raw_fd_ostream &mlirDumps() {
  static std::unique_ptr<llvm::raw_fd_ostream> S = createMlirDumpStream();
  return *S;
}

inline llvm::raw_ostream &mlirDumpsOrDbgs() {
  if (!getStrEnv("MLIR_DUMP_PATH").empty())
    return mlirDumps();
  return llvm::dbgs();
}

} // namespace mlir::triton::tools

#endif
