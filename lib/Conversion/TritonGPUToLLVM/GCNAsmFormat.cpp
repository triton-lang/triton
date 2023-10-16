/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "triton/Conversion/TritonGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream> // unify to llvm::raw_string_ostream ?

namespace mlir {
namespace triton {

GCNInstr::Operand *
GCNBuilder::newOperand(mlir::Value value, StringRef constraint,
                       std::function<std::string(int)> formatter) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formatter;
  opr->idx = oprCounter++;
  return opr;
}

GCNBuilder::Operand *GCNBuilder::newOperand(StringRef constraint) {
  // Constraint should be something like "=r"
  assert(!constraint.empty() && constraint[0] == '=');
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = constraint;
  return opr;
}

GCNBuilder::Modifier *GCNBuilder::newModifier(StringRef modifier,
                                              StringRef arg) {
  assert(!modifier.empty());
  auto *mod = newModifier();
  mod->modifier = modifier;
  mod->arg = arg;
  return mod;
}

GCNBuilder::Operand *GCNBuilder::newConstantOperand(const std::string &v) {
  argArchive.emplace_back(std::make_unique<Operand>());
  argArchive.back()->repr = [v](int idx) { return v; };
  return argArchive.back().get();
}

GCNBuilder::Operand *GCNBuilder::newConstantOperand(int v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return newConstantOperand(ss.str());
}

std::string GCNBuilder::getConstraints() const {
  auto args = getAllArgs();
  llvm::SmallVector<std::string, 4> argReprs;
  for (auto arg : args)
    argReprs.push_back(arg->constraint);
  return strJoin(argReprs, ",");
}

llvm::SmallVector<Value, 4> GCNBuilder::getAllMLIRArgs() const {
  llvm::SmallVector<Value, 4> res;
  for (auto &arg : argArchive) {
    if (!arg->isList() && arg->value)
      res.push_back(arg->value);
  }
  return res;
}

SmallVector<GCNBuilder::Operand *, 4> GCNBuilder::getAllArgs() const {
  llvm::SmallVector<Operand *, 4> res;
  for (auto &x : argArchive)
    if (!x->isList())
      res.push_back(x.get());
  return res;
}

mlir::Value GCNBuilder::launch(ConversionPatternRewriter &rewriter,
                               Location loc, Type resTy, bool hasSideEffect,
                               bool isAlignStack,
                               ArrayRef<Attribute> attrs) const {
  auto *ctx = rewriter.getContext();
  auto inlineAsm = rewriter.create<LLVM::InlineAsmOp>(
      loc, resTy, getAllMLIRArgs(), // operands
      dump(),                       // asm_string
      getConstraints(),             // constraints
      hasSideEffect,                // has_side_effects
      isAlignStack,                 // is_align_stack
      LLVM::AsmDialectAttr::get(ctx,
                                LLVM::AsmDialect::AD_ATT), // asm_dialect
      ArrayAttr::get(ctx, attrs)                           // operand_attrs
  );

  return inlineAsm.getRes();
}

std::string GCNInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return "$" + std::to_string(idx);

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return strJoin(oprs, ", ");
}

std::string GCNInstr::Modifier::dump() const {
  if (!isList())
    return to_str();

  llvm::SmallVector<std::string> mods;
  for (auto *mod : list)
    mods.push_back(mod->dump());
  return strJoin(mods, " ");
}

GCNInstr::Operand *GCNBuilder::newAddrOperand(mlir::Value addr,
                                              StringRef constraint) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [](int idx) -> std::string {
    std::stringstream ss;
    ss << "$" << idx;
    return ss.str();
  };

  return opr;
}

std::string GCNBuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &exec : executions) {
    lines.push_back(exec->dump());
  }

  return strJoin(lines, "\n\t");
}

GCNInstrExecution &GCNInstrCommon::call(ArrayRef<Operand *> oprs,
                                        ArrayRef<Modifier *> mods) {
  builder->executions.emplace_back(
      std::make_unique<GCNInstrExecution>(this, oprs, mods));
  return *builder->executions.back();
}

GCNInstrExecution &GCNInstrCommon::operator()(ArrayRef<Operand *> oprs,
                                              ArrayRef<Modifier *> mods) {
  return call(oprs, mods);
}

std::string GCNInstrExecution::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);

  std::string instrRepr = strJoin(instr->instrParts, "_");

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = strJoin(argReprs, ", ");

  llvm::SmallVector<std::string, 4> modReprs;
  for (auto *mod : mods) {
    modReprs.push_back(mod->dump());
  }

  std::string modsRepr = strJoin(modReprs, " ");
  if (!modsRepr.empty()) {
    os << instrRepr << " " << argsRepr << " " << modsRepr;
  } else {
    os << instrRepr << " " << argsRepr;
  }
  os.flush();
  return osStr;
}

SmallVector<GCNInstrExecution::Operand *>
GCNInstrExecution::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

} // namespace triton
} // namespace mlir
