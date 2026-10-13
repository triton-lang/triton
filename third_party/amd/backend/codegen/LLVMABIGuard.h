//===-- LLVMABIGuard.h - Reject an LLVM layout mismatch at compile time ---===//
//
// This library is configured separately from the LLVM it links, so nothing
// makes their assertion settings agree. They have to: LLVM's public headers
// gate *data members* on bare NDEBUG, so the two sides otherwise disagree
// about object layout.
//
//   struct MCSchedClassDesc       // llvm/MC/MCSchedule.h
//     uint32_t NameOff;           // present only when !NDEBUG
//
//   class ScheduleDAG             // llvm/CodeGen/ScheduleDAG.h
//     bool StressSched;           // a static constant when NDEBUG
//
// LLVM's own guard (llvm/Config/abi-breaking.h) does not catch this: it
// compares LLVM_ENABLE_ABI_BREAKING_CHECKS, which comes from a generated
// header and so reads the same on both sides whatever NDEBUG is. That macro is
// still the right thing to test, since it records how the linked LLVM was
// configured. A build that forces it instead of leaving the WITH_ASSERTS
// default can define TRITON_AMD_SKIP_LLVM_ABI_GUARD and match NDEBUG itself.
//
// One translation unit including this is enough; it checks properties of the
// target, not of a file.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_AMD_BACKEND_CODEGEN_LLVMABIGUARD_H
#define TRITON_AMD_BACKEND_CODEGEN_LLVMABIGUARD_H

#include "llvm/Config/abi-breaking.h"

#ifndef TRITON_AMD_SKIP_LLVM_ABI_GUARD

#if LLVM_ENABLE_ABI_BREAKING_CHECKS && defined(NDEBUG)
#error "NDEBUG is defined but the linked LLVM was built with assertions. "    \
       "LLVM's headers gate data members on NDEBUG, so this library and "     \
       "libLLVM*.a would disagree about object layout. Build this library "   \
       "with -UNDEBUG (the CMakeLists does this for you), or link an LLVM "   \
       "built without assertions."
#endif

#if !LLVM_ENABLE_ABI_BREAKING_CHECKS && !defined(NDEBUG)
#error "NDEBUG is not defined but the linked LLVM was built without "         \
       "assertions. LLVM's headers gate data members on NDEBUG, so this "     \
       "library and libLLVM*.a would disagree about object layout. Build "    \
       "this library with -DNDEBUG (the CMakeLists does this for you), or "   \
       "link an LLVM built with assertions."
#endif

#endif // TRITON_AMD_SKIP_LLVM_ABI_GUARD

#endif // TRITON_AMD_BACKEND_CODEGEN_LLVMABIGUARD_H
