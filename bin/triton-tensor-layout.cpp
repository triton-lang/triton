#include "RegisterTritonDialects.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/MLIRContext.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

// A CLI tool to print the layout of a tensor.
//
// clang-format off
// Example usage:
//
// triton-tensor-layout -l "#ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>" -t "tensor<128x256xf16>"
//
// triton-tensor-layout -i input.mlir -t "tensor<1x128x128xf16>" -o output.txt
//
// triton-tensor-layout -i input.mlir -t "tensor<1x128x128xf16>" -o output.txt -alias-names="blocked,mma" -use-hw-view
//
// An input file usually looks like:
// '''
// #mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1, 8], instrShape = [32, 32], isTransposed = false}>
// #blocked = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 16, 4], warpsPerCTA = [1, 1, 8], order = [0, 1, 2]}>
// '''
// clang-format on

//===--------------------------------------------------------------------===//
// CLI options
//===--------------------------------------------------------------------===//

static cl::OptionCategory &getPrinterCategory() {
  static cl::OptionCategory PrinterCategory(
      "Available Print Options", "Options for the tensor layout printing.");
  return PrinterCategory;
}

static cl::opt<std::string> InputFile(
    "i", cl::desc("File that contains the tensor data layout attributes"),
    cl::init(""), cl::value_desc("filename"), cl::cat(getPrinterCategory()));

static cl::opt<std::string>
    OutputFile("o", cl::desc("Output file to write the layout into"),
               cl::init(""), cl::value_desc("filename"),
               cl::cat(getPrinterCategory()));

static cl::opt<std::string>
    DataLayoutStr("l", cl::desc("Tensor data layout attribute in string"),
                  cl::value_desc("layout-string"), cl::init(""),
                  cl::cat(getPrinterCategory()));

static cl::list<std::string>
    AliasName("alias-names",
              cl::desc("A list of alias names (separated by comma) of the "
                       "layout attributes in the input file"),
              cl::value_desc("name1,name2,name3,..."), cl::CommaSeparated,
              cl::ZeroOrMore, cl::cat(getPrinterCategory()));

static cl::opt<bool> UseHWPointOfView(
    "use-hw-view",
    llvm::cl::desc(
        "Print the layout in hardware point of view. This means the output is "
        "from the warp's perspective. Otherwise, the output is from the "
        "tensor's perspective (e.g., each element maps to xxx thread)."),
    cl::init(false), cl::cat(getPrinterCategory()));

static cl::opt<std::string> TensorStr(
    "t", cl::desc("Tensor shape and element type (e.g., tensor<2x2xf32>)"),
    cl::init(""), cl::value_desc("tensor-type"), cl::cat(getPrinterCategory()));

//===--------------------------------------------------------------------===//
// Helper functions
//===--------------------------------------------------------------------===//

static mlir::Type withLayout(mlir::Type type, mlir::Attribute layout) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), layout);
  }

  auto memDescType = cast<triton::gpu::MemDescType>(type);
  return triton::gpu::MemDescType::get(
      memDescType.getShape(), memDescType.getElementType(), layout,
      memDescType.getMemorySpace(), memDescType.getMutableMemory(),
      memDescType.getAllocShape());
}

static LogicalResult layoutPrint(mlir::Type type, raw_ostream &os) {
  auto tensorOrMemDesc = cast<triton::gpu::TensorOrMemDesc>(type);
  mlir::Attribute layout = tensorOrMemDesc.getEncoding();
  if (isa<mlir::triton::gpu::DistributedEncodingTrait,
          mlir::triton::gpu::SharedEncodingTrait,
          mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr,
          mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(layout)) {
    os << triton::gpu::getLayoutStr(tensorOrMemDesc, UseHWPointOfView);
    return success();
  }

  llvm::errs() << "Unsupported tensor layout attribute: "
               << tensorOrMemDesc.getEncoding() << "\n";
  return failure();
}

static LogicalResult printLayoutFromFile(MLIRContext *context,
                                         StringRef filename,
                                         ArrayRef<std::string> names,
                                         mlir::Type tensorTy,
                                         raw_string_ostream &ss) {
  if (filename.empty())
    return success();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  ParserConfig config(context);
  auto asmState = AsmParserState();

  Block parsedIR;
  if (failed(parseAsmSourceFile(sourceMgr, &parsedIR, config, &asmState))) {
    llvm::errs() << "Fail to parse the input file: " << filename << "\n";
    return failure();
  }

  auto printLambda = [&](StringRef name, mlir::Attribute attr) {
    ss << "Print layout attribute: #" << name << " = " << attr << "\n";

    return layoutPrint(withLayout(tensorTy, attr), ss);
  };

  if (names.empty())
    // If no alias name is given, we print all layout attributes in the file.
    for (const auto &def : asmState.getAttributeAliasDefs()) {
      if (failed(printLambda(def.name, def.value)))
        return failure();
    }
  else {
    // Print the layout attributes with the given alias names.
    for (const auto &alias : names) {
      auto def = asmState.getAttributeAliasDef(alias);
      if (!def) {
        llvm::errs() << "Can't find the layout attribute: " << alias << "\n";
        return failure();
      }

      if (failed(printLambda(alias, def->value)))
        return failure();

      ss << "\n";
    }
  }

  return success();
}

static LogicalResult printLayoutFromString(MLIRContext *context,
                                           StringRef layoutAttrStr,
                                           mlir::Type tensorTy,
                                           raw_string_ostream &ss) {
  if (layoutAttrStr.empty())
    return success();

  mlir::Attribute layout = parseAttribute(layoutAttrStr, context);
  if (!layout) {
    llvm::errs() << "Invalid layout attribute: " << layoutAttrStr << "\n";
    return failure();
  }

  ss << "Print layout attribute: " << layout << "\n";

  return layoutPrint(withLayout(tensorTy, layout), ss);
}

//===--------------------------------------------------------------------===//
// Main entry point
//===--------------------------------------------------------------------===//

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(getPrinterCategory());
  cl::ParseCommandLineOptions(argc, argv, "tensor layout printer\n");

  DialectRegistry registry;
  registerTritonDialects(registry);

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  if (TensorStr.empty()) {
    llvm::errs() << "Must specify the tensor type argument\n";
    return 1;
  }

  mlir::Type parsedTy = parseType(TensorStr, &ctx);
  if (!parsedTy) {
    llvm::errs() << "Fail to parse the tensor type argument: " << TensorStr
                 << "\n";
    return 1;
  }

  if (!isa<RankedTensorType, triton::gpu::MemDescType>(parsedTy)) {
    llvm::errs() << "Invalid tensor or memdesc type argument: " << TensorStr
                 << "\n";
    return 1;
  }

  std::string storage;
  raw_string_ostream ss(storage);

  if (failed(printLayoutFromFile(&ctx, InputFile, AliasName, parsedTy, ss)))
    return 1;

  if (failed(printLayoutFromString(&ctx, DataLayoutStr, parsedTy, ss)))
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
