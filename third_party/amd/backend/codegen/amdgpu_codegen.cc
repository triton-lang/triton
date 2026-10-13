#include "LLVMABIGuard.h"

#include "triton/Tools/LLVMOptions.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
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
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Error.h"
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

using mlir::triton::tools::ScopedLLVMOptions;

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

void enableFPContraction(llvm::Module &module) {
  for (llvm::Function &function : module)
    for (llvm::Instruction &instruction : llvm::instructions(function))
      if (instruction.getOpcode() == llvm::Instruction::FAdd ||
          instruction.getOpcode() == llvm::Instruction::FSub ||
          instruction.getOpcode() == llvm::Instruction::FMul)
        instruction.setHasAllowContract(true);
}

// LLVM 5bf967cb132b moved named barriers from the LDS address space into their
// own address space, and AutoUpgrade does not convert the legacy form
// emitted by Triton's LLVM. Retype barrier handles by following them forward
// from the named-barrier globals through direct helper calls and SSA
// forwarding.
// TODO: Remove once Triton's LLVM (cmake/llvm-hash.txt) includes 5bf967cb132b
// and ConvertWarpSpecializeToLLVM creates named barriers in the new address
// space.
llvm::Error upgradeLegacyNamedBarriers(llvm::Module &module) {
  const llvm::Intrinsic::ID barrierIntrinsics[] = {
      llvm::Intrinsic::amdgcn_s_barrier_init,
      llvm::Intrinsic::amdgcn_s_barrier_join,
      llvm::Intrinsic::amdgcn_s_barrier_signal_var,
      llvm::Intrinsic::amdgcn_s_get_named_barrier_state,
      llvm::Intrinsic::amdgcn_s_wakeup_barrier};
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *barrierType =
      llvm::Intrinsic::getType(context, llvm::Intrinsic::amdgcn_s_barrier_join)
          ->getParamType(0);

  llvm::SmallSetVector<llvm::Value *, 16> handles;
  for (llvm::GlobalVariable &global : module.globals()) {
    auto *type = llvm::dyn_cast<llvm::TargetExtType>(global.getValueType());
    if (type && type->getName() == "amdgcn.named.barrier" &&
        global.getAddressSpace() == llvm::AMDGPUAS::LOCAL_ADDRESS)
      handles.insert(&global);
  }

  auto unsupported = [](const llvm::User *user) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    stream << "unsupported use of a legacy named barrier:\n";
    user->print(stream);
    return llvm::createStringError(message);
  };

  // Collect every handle before retyping anything.
  llvm::SmallSetVector<llvm::Function *, 8> helpers;
  for (size_t i = 0; i < handles.size(); ++i) {
    for (llvm::Use &use : handles[i]->uses()) {
      llvm::User *user = use.getUser();
      if (llvm::isa<llvm::PHINode, llvm::SelectInst, llvm::FreezeInst>(user)) {
        handles.insert(user);
        continue;
      }
      auto *call = llvm::dyn_cast<llvm::CallBase>(user);
      llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
      if (!callee || !call->isArgOperand(&use))
        return unsupported(user);
      unsigned argNo = call->getArgOperandNo(&use);
      if (argNo == 0 &&
          llvm::is_contained(barrierIntrinsics, callee->getIntrinsicID()))
        continue;
      if (callee->isDeclaration() || argNo >= callee->arg_size())
        return unsupported(user);
      handles.insert(callee->getArg(argNo));
      helpers.insert(callee);
    }
  }
  // Helpers (functions that receive a barrier handle) are rewritten below, so
  // each of their uses must be a direct call whose signature matches the
  // helper's original signature.
  for (llvm::Function *helper : helpers)
    for (llvm::Use &use : helper->uses()) {
      auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
      if (!call || !call->isCallee(&use) ||
          call->getFunctionType() != helper->getFunctionType())
        return unsupported(use.getUser());
    }

  for (llvm::Value *handle : handles)
    handle->mutateType(barrierType);

  // A function's signature cannot be changed in place, so move each helper's
  // arguments and body into a new function whose signature matches the
  // retyped arguments.
  for (llvm::Function *helper : helpers) {
    llvm::SmallVector<llvm::Type *> params;
    for (llvm::Argument &arg : helper->args())
      params.push_back(arg.getType());
    llvm::Function *replacement = llvm::Function::Create(
        llvm::FunctionType::get(helper->getReturnType(), params,
                                helper->isVarArg()),
        helper->getLinkage(), helper->getAddressSpace());
    module.getFunctionList().insert(helper->getIterator(), replacement);
    replacement->copyAttributesFrom(helper);
    replacement->setComdat(helper->getComdat());
    replacement->copyMetadata(helper, 0);
    replacement->stealArgumentListFrom(*helper);
    replacement->splice(replacement->begin(), helper);
    for (llvm::Use &use : llvm::make_early_inc_range(helper->uses()))
      llvm::cast<llvm::CallBase>(use.getUser())->setCalledFunction(replacement);
    replacement->takeName(helper);
    helper->eraseFromParent();
  }

  for (llvm::Intrinsic::ID id : barrierIntrinsics) {
    llvm::Function *legacy =
        llvm::Intrinsic::getDeclarationIfExists(&module, id);
    if (!legacy ||
        legacy->getFunctionType() == llvm::Intrinsic::getType(context, id))
      continue;
    legacy->setName("");
    llvm::Function *declaration =
        llvm::Intrinsic::getOrInsertDeclaration(&module, id);
    for (llvm::Use &use : llvm::make_early_inc_range(legacy->uses())) {
      auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
      if (!call || !call->isCallee(&use))
        return unsupported(use.getUser());
      call->setCalledFunction(declaration);
    }
    legacy->eraseFromParent();
  }
  return llvm::Error::success();
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

  std::vector<ScopedLLVMOptions::Setting> settings;
  if (options->dumpIR)
    settings.emplace_back("print-after-all", "true");
  if (options->enableTiming) {
    settings.emplace_back("time-passes", "true");
    settings.emplace_back("time-passes-per-run", "true");
  }

  if (options->flags && options->flags[0]) {
    llvm::SmallVector<llvm::StringRef, 4> flags;
    llvm::StringRef(options->flags).split(flags, ',');
    for (llvm::StringRef flag : flags)
      if (!flag.empty())
        settings.emplace_back(flag.str(), "true");
  }

  if (options->disabledPasses && options->disabledPasses[0]) {
    llvm::SmallVector<llvm::StringRef, 4> disabledPasses;
    llvm::StringRef(options->disabledPasses).split(disabledPasses, ',');
    for (llvm::StringRef pass : disabledPasses)
      if (!pass.empty())
        settings.emplace_back(pass.str(), "true");
  }
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module;
  {
    // UpgradeDebugInfo can abort on legacy named-barrier signatures while
    // parsing. Defer it until after the barrier upgrade and verifyModule, so
    // verification errors are returned through fail() instead of aborting the
    // process.
    std::vector<ScopedLLVMOptions::Setting> parseSettings = settings;
    parseSettings.emplace_back("disable-auto-upgrade-debug-info", "true");
    ScopedLLVMOptions parseScope(parseSettings);

    llvm::SMDiagnostic diagnostic;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(llvmIR, llvmIRSize), "triton-amd-codegen", false);
    module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
    if (!module) {
      std::string message;
      llvm::raw_string_ostream stream(message);
      diagnostic.print("triton-amd-codegen", stream);
      return fail(message, error);
    }
    if (llvm::Error upgradeError = upgradeLegacyNamedBarriers(*module))
      return fail(llvm::toString(std::move(upgradeError)), error);

    std::string message;
    llvm::raw_string_ostream stream(message);
    bool brokenDebugInfo = false;
    if (llvm::verifyModule(*module, &stream, &brokenDebugInfo))
      return fail("invalid LLVM IR:\n" + message, error);
  }

  ScopedLLVMOptions optionScope(settings);
  llvm::UpgradeDebugInfo(*module);

  // LLVM no longer supports globally enabling FP contraction through
  // TargetOptions. Preserve enableFPFusion by expressing that permission on
  // the operations consumed by the independently pinned code generator.
  if (options->enableFPFusion)
    enableFPContraction(*module);

  module->setTargetTriple(llvm::Triple(options->triple));
  std::string targetError;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      module->getTargetTriple(), targetError);
  if (!target)
    return fail(targetError, error);

  llvm::TargetOptions targetOptions;
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

  ScopedLLVMOptions optionScope({});
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
