#pragma once

#include <nanobind/nanobind.h>
#include <string>

// Top-level entry point behind `libtriton.extend_with(path)`. Loads a Triton
// extension once and registers all of its dialects, custom operations, and
// passes by walking the given top-level module (`ir.builder`, `passes`, ...).
// Defined in `ir.cc`.
void extendTritonWith(nanobind::module_ m, const std::string &path);
