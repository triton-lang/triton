#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#if defined(_WIN32)
#define TRITON_AMD_EXPORT __declspec(dllexport)
#else
#define TRITON_AMD_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr uint32_t codegenABIVersion = 1;

struct CodegenOptions {
  uint32_t abiVersion;
  const char *triple;
  const char *processor;
  const char *features;
  const char *abi;
  const char *flags;
  const char *disabledPasses;
  uint8_t enableFPFusion;
  uint8_t disableOptimization;
  uint8_t canonicalizeGEP;
  uint8_t dumpIR;
  uint8_t enableTiming;
};

std::once_flag targetInitialization;

bool setLLVMOption(llvm::StringRef name, bool value) {
  auto options = llvm::cl::getRegisteredOptions();
  auto option = options.find(name);
  if (option == options.end())
    return false;

  auto *typedOption = static_cast<llvm::cl::opt<bool> *>(option->second);
  bool previous = typedOption->getValue();
  option->second->addOccurrence(1, name, value ? "true" : "false");
  return previous;
}

char *copyString(llvm::StringRef value) {
  auto *result = static_cast<char *>(std::malloc(value.size() + 1));
  if (!result)
    return nullptr;
  std::memcpy(result, value.data(), value.size());
  result[value.size()] = '\0';
  return result;
}

int fail(const std::string &message, char **error) {
  if (error)
    *error = copyString(message);
  return 1;
}

void initializeTarget() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  llvm::PassRegistry &registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(registry);
  llvm::initializeCodeGen(registry);
  llvm::initializeLoopStrengthReducePass(registry);
  llvm::initializePostInlineEntryExitInstrumenterPass(registry);
  llvm::initializeUnreachableBlockElimLegacyPassPass(registry);
  llvm::initializeConstantHoistingLegacyPassPass(registry);
  llvm::initializeScalarOpts(registry);
  llvm::initializeIPO(registry);
  llvm::initializeVectorization(registry);
  llvm::initializeScalarizeMaskedMemIntrinLegacyPassPass(registry);
  llvm::initializeTransformUtils(registry);

  llvm::parallel::strategy = llvm::hardware_concurrency(1);
}

} // namespace

extern "C" TRITON_AMD_EXPORT int
triton_amdgpu_compile(const char *llvmIR, size_t llvmIRSize,
                      const CodegenOptions *options, char **amdgcn,
                      size_t *amdgcnSize, char **error) {
  if (error)
    *error = nullptr;
  if (amdgcn)
    *amdgcn = nullptr;
  if (amdgcnSize)
    *amdgcnSize = 0;

  if (!llvmIR || !options || !amdgcn || !amdgcnSize)
    return fail("invalid AMD code-generation arguments", error);
  if (options->abiVersion != codegenABIVersion)
    return fail("incompatible AMD code-generation ABI", error);
  if (!options->triple || !options->processor || !options->features ||
      !options->abi)
    return fail("incomplete AMD code-generation target options", error);

  std::call_once(targetInitialization, initializeTarget);

  if (options->dumpIR)
    setLLVMOption("print-after-all", true);

  if (options->flags && options->flags[0]) {
    llvm::SmallVector<llvm::StringRef, 4> flags;
    llvm::StringRef(options->flags).split(flags, ',');
    for (llvm::StringRef flag : flags)
      if (!flag.empty())
        setLLVMOption(flag, true);
  }

  if (options->disabledPasses && options->disabledPasses[0]) {
    llvm::SmallVector<llvm::StringRef, 4> disabledPasses;
    llvm::StringRef(options->disabledPasses).split(disabledPasses, ',');
    for (llvm::StringRef pass : disabledPasses)
      if (!pass.empty())
        setLLVMOption(pass, true);
  }

  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(llvmIR, llvmIRSize), "triton-amd-codegen", false);
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("triton-amd-codegen", stream);
    return fail(message, error);
  }

  module->setTargetTriple(llvm::Triple(options->triple));
  std::string targetError;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      module->getTargetTriple(), targetError);
  if (!target)
    return fail(targetError, error);

  llvm::TargetOptions targetOptions;
  if (options->enableFPFusion)
    targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  targetOptions.TrapUnreachable = true;
  targetOptions.MCOptions.AsmVerbose = true;
  targetOptions.MCOptions.PreserveAsmComments = true;
  targetOptions.MCOptions.ABIName = options->abi;

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      module->getTargetTriple(), options->processor, options->features,
      targetOptions, llvm::Reloc::PIC_, std::nullopt,
      options->disableOptimization ? llvm::CodeGenOptLevel::None
                                   : llvm::CodeGenOptLevel::Aggressive));
  if (!machine)
    return fail("failed to create AMD target machine", error);
  module->setDataLayout(machine->createDataLayout());

  for (llvm::Function &function : module->functions())
    if (!function.hasFnAttribute(llvm::Attribute::NoInline))
      function.addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::legacy::PassManager inlinePasses;
  inlinePasses.add(llvm::createTargetTransformInfoWrapperPass(
      machine->getTargetIRAnalysis()));
  inlinePasses.add(llvm::createAlwaysInlinerLegacyPass());
  inlinePasses.add(llvm::createVerifierPass());

  if (options->enableTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  inlinePasses.run(*module);

  if (options->canonicalizeGEP && !options->disableOptimization) {
    llvm::legacy::PassManager cleanup;
    cleanup.add(llvm::createTargetTransformInfoWrapperPass(
        machine->getTargetIRAnalysis()));
    cleanup.add(llvm::createSeparateConstOffsetFromGEPPass());
    cleanup.add(llvm::createEarlyCSEPass());
    cleanup.run(*module);
  }

  std::string assembly;
  {
    llvm::raw_string_ostream output(assembly);
    llvm::buffer_ostream bufferedOutput(output);
    llvm::legacy::PassManager codegen;
    if (machine->addPassesToEmitFile(codegen, bufferedOutput, nullptr,
                                     llvm::CodeGenFileType::AssemblyFile))
      return fail("AMD target cannot emit AMDGCN assembly", error);
    codegen.run(*module);
  }

  if (options->enableTiming) {
    llvm::SmallString<0> timings;
    llvm::raw_svector_ostream stream(timings);
    llvm::reportAndResetTimings(&stream);
    llvm::dbgs() << stream.str();
  }

  *amdgcn = copyString(assembly);
  if (!*amdgcn)
    return fail("failed to allocate AMDGCN assembly result", error);
  *amdgcnSize = assembly.size();
  return 0;
}

extern "C" TRITON_AMD_EXPORT int
triton_amdgpu_assemble(const char *assembly, size_t assemblySize,
                       const char *triple, const char *processor,
                       const char *features, char **object, size_t *objectSize,
                       char **error) {
  if (error)
    *error = nullptr;
  if (object)
    *object = nullptr;
  if (objectSize)
    *objectSize = 0;

  if (!assembly || !triple || !processor || !features || !object || !objectSize)
    return fail("invalid AMD assembly arguments", error);

  std::call_once(targetInitialization, initializeTarget);

  llvm::Triple targetTriple(triple);
  std::string targetError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, targetError);
  if (!target)
    return fail("target lookup error: " + targetError, error);

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(assembly, assemblySize),
                                       "triton-amdgpu-assembler", false),
      llvm::SMLoc());

  const llvm::MCTargetOptions options;
  std::unique_ptr<llvm::MCRegisterInfo> registers(
      target->createMCRegInfo(targetTriple));
  std::unique_ptr<llvm::MCAsmInfo> asmInfo(
      target->createMCAsmInfo(*registers, targetTriple, options));
  std::unique_ptr<llvm::MCSubtargetInfo> subtarget(
      target->createMCSubtargetInfo(targetTriple, processor, features));

  llvm::MCContext context(targetTriple, *asmInfo, *registers, *subtarget,
                          &sourceManager);
  std::unique_ptr<llvm::MCObjectFileInfo> objectInfo(
      target->createMCObjectFileInfo(context, /*PIC=*/false,
                                     /*LargeCodeModel=*/false));
  context.setObjectFileInfo(objectInfo.get());

  llvm::SmallString<128> workingDirectory;
  if (!llvm::sys::fs::current_path(workingDirectory))
    context.setCompilationDir(workingDirectory);

  llvm::SmallVector<char, 0> result;
  llvm::raw_svector_ostream output(result);
  std::unique_ptr<llvm::MCInstrInfo> instructions(target->createMCInstrInfo());
  std::unique_ptr<llvm::MCCodeEmitter> emitter(
      target->createMCCodeEmitter(*instructions, context));
  std::unique_ptr<llvm::MCAsmBackend> backend(
      target->createMCAsmBackend(*subtarget, *registers, options));
  std::unique_ptr<llvm::MCObjectWriter> writer(
      backend->createObjectWriter(output));
  std::unique_ptr<llvm::MCStreamer> streamer(target->createMCObjectStreamer(
      targetTriple, context, std::move(backend), std::move(writer),
      std::move(emitter), *subtarget));

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(sourceManager, context, *streamer, *asmInfo));
  std::unique_ptr<llvm::MCTargetAsmParser> targetParser(
      target->createMCAsmParser(*subtarget, *parser, *instructions));
  if (!targetParser)
    return fail("AMD assembler initialization error", error);

  parser->setTargetParser(*targetParser);
  if (parser->Run(/*NoInitialTextSection=*/false))
    return fail("AMD assembler rejected generated AMDGCN", error);

  *object = copyString(llvm::StringRef(result.data(), result.size()));
  if (!*object)
    return fail("failed to allocate AMD object result", error);
  *objectSize = result.size();
  return 0;
}

extern "C" TRITON_AMD_EXPORT void triton_amdgpu_free(void *pointer) {
  std::free(pointer);
}

extern "C" TRITON_AMD_EXPORT const char *triton_amdgpu_revision() {
  return TRITON_AMD_LLVM_REVISION;
}
