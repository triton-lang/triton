#include "triton/Conversion/TritonGPUToLLVM/PtxAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream> // unify to llvm::raw_string_ostream ?

namespace mlir {
namespace triton {

// TODO(Superjomn) Move to a global utility file?
std::string strJoin(llvm::ArrayRef<std::string> strs,
                    llvm::StringRef delimiter) {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);
  for (size_t i = 0; !strs.empty() && i < strs.size() - 1; ++i)
    os << strs[i] << delimiter;
  if (!strs.empty())
    os << strs.back();
  os.flush();
  return osStr;
}

PTXInstr::Operand *
PTXBuilder::newOperand(mlir::Value value, StringRef constraint,
                       std::function<std::string(int)> formatter) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formatter;
  opr->idx = oprCounter++;
  return opr;
}

PTXBuilder::Operand *PTXBuilder::newOperand(StringRef constraint) {
  // Constraint should be something like "=r"
  assert(!constraint.empty() && constraint[0] == '=');
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = constraint;
  return opr;
}

PTXBuilder::Operand *PTXBuilder::newConstantOperand(const std::string &v) {
  argArchive.emplace_back(std::make_unique<Operand>());
  argArchive.back()->repr = [v](int idx) { return v; };
  return argArchive.back().get();
}

PTXBuilder::Operand *PTXBuilder::newConstantOperand(int64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return newConstantOperand(ss.str());
}

std::string PTXBuilder::getConstraints() const {
  auto args = getAllArgs();
  llvm::SmallVector<std::string, 4> argReprs;
  for (auto arg : args)
    argReprs.push_back(arg->constraint);
  return strJoin(argReprs, ",");
}

llvm::SmallVector<Value, 4> PTXBuilder::getAllMLIRArgs() const {
  llvm::SmallVector<Value, 4> res;
  for (auto &arg : argArchive) {
    if (!arg->isList() && arg->value)
      res.push_back(arg->value);
  }
  return res;
}

SmallVector<PTXBuilder::Operand *, 4> PTXBuilder::getAllArgs() const {
  llvm::SmallVector<Operand *, 4> res;
  for (auto &x : argArchive)
    if (!x->isList())
      res.push_back(x.get());
  return res;
}

mlir::Value PTXBuilder::launch(ConversionPatternRewriter &rewriter,
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

std::string PTXInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return "$" + std::to_string(idx);

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return "{ " + strJoin(oprs, ", ") + " }";
}

PTXInstr::Operand *PTXBuilder::newAddrOperand(mlir::Value addr,
                                              StringRef constraint, int off) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [off](int idx) -> std::string {
    std::stringstream ss;
    ss << "[ $" << idx << " + " << off << " ]";
    return ss.str();
  };

  return opr;
}

std::string PTXBuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &exec : executions) {
    lines.push_back(exec->dump());
  }

  return strJoin(lines, "\n\t");
}

PTXInstrExecution &PTXInstrCommon::call(ArrayRef<Operand *> oprs,
                                        bool onlyAttachMLIRArgs) {
  if (onlyAttachMLIRArgs) {
    // Nearly impossible to make the $0,$1 in two PTX code snippets to point to
    // the same MLIR values in onlyAttachMLIRArgs mode.
    assert(builder->executions.empty() &&
           "builder can only hold a single execution when onlyAttachMIIRArgs "
           "is true.");
    builder->reorderArgArchive(oprs);
  }

  builder->executions.emplace_back(
      std::make_unique<PTXInstrExecution>(this, oprs, onlyAttachMLIRArgs));

  return *builder->executions.back();
}

PTXInstrExecution &PTXInstrCommon::operator()(ArrayRef<Operand *> oprs,
                                              bool onlyAttachMLIRArgs) {
  return call(oprs, onlyAttachMLIRArgs);
}

std::string PTXInstrExecution::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);

  std::string instrRepr = strJoin(instr->instrParts, ".");
  if (onlyAttachMLIRArgs)
    return instrRepr;

  if (pred) {
    if (!pred->repr)
      os << "@" << pred->dump() << " ";
    else
      os << pred->repr(pred->idx) << " ";
  }

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = strJoin(argReprs, ", ");

  os << instrRepr << " " << argsRepr << ";";
  os.flush();
  return osStr;
}

SmallVector<PTXInstrExecution::Operand *>
PTXInstrExecution::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

PTXInstr &PTXInstr::global() {
  o("global");
  return *this;
}

PTXInstr &PTXInstr::shared() {
  o("shared");
  return *this;
}

PTXInstr &PTXInstr::v(int vecWidth, bool predicate) {
  if (vecWidth > 1) {
    o("v" + std::to_string(vecWidth), predicate);
  }
  return *this;
}

PTXInstr &PTXInstr::b(int width) {
  o("b" + std::to_string(width));
  return *this;
}

} // namespace triton
} // namespace mlir
