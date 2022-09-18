#ifndef _TRITON_CODE_GEN_EXTERN_LIB_H_
#define _TRITON_CODE_GEN_EXTERN_LIB_H_

#include <memory>
#include <string>
#include <map>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

namespace triton {
namespace codegen {

///
/// \brief ExternLib is a class that represents a library of external functions.
///
class ExternLib {
 public:
  ExternLib(const std::string &name, const std::string &path)
      : name_(name), path_(path) {}

  virtual ~ExternLib() = default;

  virtual const std::string &name() const { return name_; }

  virtual const std::string &path() const { return path_; }

  ///
  /// \brief Load the library and return the module.
  ///
  std::unique_ptr<llvm::Module> load(llvm::LLVMContext &ctx);

  ///
  /// \brief Link the module into the given module.
  ///
  void link(std::unique_ptr<llvm::Module> &llvm,
            std::unique_ptr<llvm::Module> &mod);

  ///
  /// \brief Run load, link, and opt on the module.
  ///
  virtual void install(llvm::LLVMContext &ctx,
                       std::unique_ptr<llvm::Module> &llvm) {
    auto mod = load(ctx);
    link(llvm, mod);
    opt(ctx, llvm);
  }

  ///
  /// \brief Run opt on the module.
  ///
  virtual void opt(llvm::LLVMContext &ctx,
                   std::unique_ptr<llvm::Module> &llvm) = 0;

 private:
  std::string name_;
  std::string path_;
};

///
/// \brief ExternLibMap is a map of ExternLibs from their names to their paths.
///
typedef std::map<std::string, std::unique_ptr<ExternLib>> ExternLibMap;

///
/// \brief Concrete class for NVIDIA's libdevice library.
///
class LibDevice final : public ExternLib {
 public:
  LibDevice(const std::string &name, const std::string &path)
      : ExternLib(name, path) {}

  virtual ~LibDevice() = default;

  virtual void opt(llvm::LLVMContext &ctx,
                   std::unique_ptr<llvm::Module> &llvm) override;
};

///
/// \brief Create an ExternLib instance based on the name and path.
///
std::unique_ptr<ExternLib> create_extern_lib(const std::string &lib_name,
                                             const std::string &lib_path);

}  // namespace codegen
}  // namespace triton

#endif
