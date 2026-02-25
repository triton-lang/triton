#ifndef TRITON_TOOLS_SYS_DUMP_HPP
#define TRITON_TOOLS_SYS_DUMP_HPP

#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::tools {

inline llvm::raw_fd_ostream &mlirDumps() {
  std::error_code EC;
  static llvm::raw_fd_ostream S(getStrEnv("MLIR_DUMP_PATH"), EC,
                                llvm::sys::fs::CD_CreateAlways);
  assert(!EC && "failed to open MLIR_DUMP_PATH");
  return S;
}

inline llvm::raw_ostream &mlirDumpsOrDbgs() {
  if (!getStrEnv("MLIR_DUMP_PATH").empty())
    return mlirDumps();
  return llvm::dbgs();
}

} // namespace mlir::triton::tools

#endif
