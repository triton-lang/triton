#ifndef TRITON_TOOLS_LLVMDIAGNOSTICCAPTURE_H
#define TRITON_TOOLS_LLVMDIAGNOSTICCAPTURE_H

#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::triton::tools {

class LLVMDiagnosticCapture {
  llvm::LLVMContext &context;
  llvm::DiagnosticHandler::DiagnosticHandlerTy previousCallback;
  void *previousContext;
  std::string message;
  bool sawError = false;

  static void handleDiagnostic(const llvm::DiagnosticInfo *diagnostic,
                               void *context) {
    auto *capture = static_cast<LLVMDiagnosticCapture *>(context);
    llvm::DiagnosticSeverity severity = diagnostic->getSeverity();
    if (severity == llvm::DS_Error)
      capture->sawError = true;
    if (severity == llvm::DS_Remark || severity == llvm::DS_Note)
      return;
    llvm::raw_string_ostream stream(capture->message);
    stream << llvm::LLVMContext::getDiagnosticMessagePrefix(severity) << ": ";
    llvm::DiagnosticPrinterRawOStream printer(stream);
    diagnostic->print(printer);
    stream << '\n';
  }

public:
  explicit LLVMDiagnosticCapture(llvm::LLVMContext &context)
      : context(context),
        previousCallback(context.getDiagnosticHandlerCallBack()),
        previousContext(context.getDiagnosticContext()) {
    context.setDiagnosticHandlerCallBack(handleDiagnostic, this,
                                         /*RespectFilters=*/false);
  }

  ~LLVMDiagnosticCapture() {
    context.setDiagnosticHandlerCallBack(previousCallback, previousContext);
  }

  LLVMDiagnosticCapture(const LLVMDiagnosticCapture &) = delete;
  LLVMDiagnosticCapture &operator=(const LLVMDiagnosticCapture &) = delete;

  bool hasErrors() const { return sawError; }
  const std::string &getMessage() const { return message; }
};

} // namespace mlir::triton::tools

#endif // TRITON_TOOLS_LLVMDIAGNOSTICCAPTURE_H
