#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/MLIRContext.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

//===--------------------------------------------------------------------===//
// CLI options
//===--------------------------------------------------------------------===//

cl::OptionCategory PrinterCategory("Available Print Options",
                                   "Options for the tensor layout printing.");

static cl::opt<std::string> InputFile(
    "i", cl::desc("File name that contains the tensor data layout attribute"),
    cl::init(""), cl::value_desc("filename"), cl::cat(PrinterCategory));

static cl::opt<std::string>
    OutputFile("o", cl::desc("Output filename to write the layout"),
               cl::init(""), cl::value_desc("filename"),
               cl::cat(PrinterCategory));

static cl::opt<std::string>
    DataLayoutStr("l", cl::desc("Tensor data layout attribute in string"),
                  cl::value_desc("layout-string"), cl::init(""),
                  cl::cat(PrinterCategory));

static cl::list<std::string>
    AliasName("alias-names",
              cl::desc("A list of alias names (separated by comma) of the "
                       "layout attributes in the input file"),
              cl::value_desc("names"), cl::cat(PrinterCategory));

static cl::opt<bool> useHWPointOfView(
    "use-hw-view", llvm::cl::desc("Print the layout in hardware point of view"),
    cl::init(false), cl::cat(PrinterCategory));

static cl::opt<std::string> TensorStr(
    "t", cl::desc("Tensor shape and element type (e.g., tensor<2x2xf32>)"),
    cl::init(""), cl::value_desc("tensor-type"), cl::cat(PrinterCategory));

//===--------------------------------------------------------------------===//
// Helper functions
//===--------------------------------------------------------------------===//

LogicalResult layoutPrint(RankedTensorType tensorType, raw_ostream &os) {
  StringRef dialectName = tensorType.getEncoding().getDialect().getNamespace();

  // Dispatch to the corresponding dialect helper to print the layout.
  if (dialectName == "triton_gpu") {
    os << triton::gpu::getLayoutStr(tensorType, useHWPointOfView);
    return success();
  }

  llvm::errs() << "Unsupported tensor layout attribute: "
               << tensorType.getEncoding() << "\n";
  return failure();
}

LogicalResult printLayoutFromFile(MLIRContext *context, StringRef filename,
                                  ArrayRef<std::string> names,
                                  TensorType tensorTy, raw_string_ostream &ss) {
  if (filename.empty())
    return success();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  ParserConfig config(context);
  auto asmState = AsmParserState();

  Block parsedIR;
  if (failed(parseAsmSourceFile(sourceMgr, &parsedIR, config, &asmState))) {
    llvm::errs() << "Fail to parse the file: " << filename << "\n";
    return failure();
  }

  auto printLambda = [&](StringRef name, Attribute attr) {
    ss << "Print layout attribute: #" << name << " = " << attr << "\n";

    auto rankedTensorTy = RankedTensorType::get(
        tensorTy.getShape(), tensorTy.getElementType(), attr);

    return layoutPrint(rankedTensorTy, ss);
  };

  if (names.empty())
    // If no alias name is given, we print all layout attributes in the file.
    for (auto def : asmState.getAttributeAliasDefs()) {
      if (failed(printLambda(def.name, def.value)))
        return failure();
    }
  else {
    // Print the layout attributes with the given alias names.
    for (auto alias : names) {
      auto def = asmState.getAttributeAliasDef(alias);
      if (!def) {
        llvm::errs() << "Can't find the layout attribute: " << alias << "\n";
        return failure();
      }

      if (failed(printLambda(alias, def->value)))
        return failure();
    }
  }

  return success();
}

LogicalResult printLayoutFromString(MLIRContext *context,
                                    StringRef layoutAttrStr,
                                    TensorType tensorTy,
                                    raw_string_ostream &ss) {
  if (layoutAttrStr.empty())
    return success();

  Attribute layout = parseAttribute(layoutAttrStr, context);
  if (!layout) {
    llvm::errs() << "Invalid layout attribute: " << layoutAttrStr << "\n";
    return failure();
  }

  auto rankedTensorTy = RankedTensorType::get(
      tensorTy.getShape(), tensorTy.getElementType(), layout);

  ss << "Print layout attribute: #" << layout << "\n";

  return layoutPrint(rankedTensorTy, ss);
}

//===--------------------------------------------------------------------===//
// Main entry point
//===--------------------------------------------------------------------===//

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(PrinterCategory);
  cl::ParseCommandLineOptions(argc, argv, "tensor layout printer\n");

  DialectRegistry registry;
  // Register all dialects that can print tensor layout.
  registry.insert<triton::gpu::TritonGPUDialect>();

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  if (TensorStr.empty()) {
    llvm::errs() << "Must specify tensor type argument\n";
    return 1;
  }

  TensorType tensorType = dyn_cast<TensorType>(parseType(TensorStr, &ctx));
  if (!tensorType) {
    llvm::errs() << "Invalid tensor type argument: " << TensorStr << "\n";
    return 1;
  }

  std::string storage;
  raw_string_ostream ss(storage);

  if (failed(printLayoutFromString(&ctx, DataLayoutStr, tensorType, ss)))
    return 1;

  if (failed(printLayoutFromFile(&ctx, InputFile, AliasName, tensorType, ss)))
    return 1;

  if (OutputFile.empty()) {
    llvm::outs() << ss.str();
  } else {
    std::error_code ec;
    llvm::raw_fd_ostream outFs(OutputFile, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "Error: " << ec.message() << " : unable to open "
                   << OutputFile << " for output\n";
      return 1;
    }
    outFs << ss.str();
    outFs.close();
  }

  return 0;
}
