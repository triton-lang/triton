#pragma once

// Clang's CUDA runtime wrapper force-includes this header to work around
// conflicting builtin-variable redeclarations in the real CURAND header.
// GSan does not use CURAND, so an empty shim is sufficient for device-only
// LLVM IR generation against the vendored CUDA headers.
