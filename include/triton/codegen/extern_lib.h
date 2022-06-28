#ifndef _TRITON_CODE_GEN_EXTERN_LIB_H_
#define _TRITON_CODE_GEN_EXTERN_LIB_H_

#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace triton {

namespace codegen {

class ExternLib {
 public:
  ExternLib(const std::string &name, const std::string &path)
      : name_(name), path_(path) {}

  virtual ~ExternLib() = default;

  virtual void link(llvm::LLVMContext &ctx,
                    std::unique_ptr<llvm::Module> &llvm) = 0;

 private:
  std::string name_;
  std::string path_;
};

class LibDevice final : public ExternLib {
 public:
  LibDevice(const std::string &name, const std::string &path)
      : ExternLib(name, path) {}

  virtual ~LibDevice() = default;

  virtual void link(llvm::LLVMContext &ctx,
                    std::unique_ptr<llvm::Module> &llvm);
};

}  // namespace code_gen

}  // namespace triton

#endif