#include "triton/Conversion/TritonGPUToLLVM/PtxAsmFormat.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream> // unify to llvm::raw_string_ostream ?

namespace mlir {
namespace triton {

// TODO(Superjomn) Move to a global utility file?
std::string strJoin(llvm::ArrayRef<std::string> strs,
                    llvm::StringRef delimiter) {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);
  for (size_t i = 0; !strs.empty() && i < strs.size() - 1; i++)
    os << strs[i] << delimiter;
  if (!strs.empty())
    os << strs.back();
  os.flush();
  return osStr;
}

PTXInstr::Operand *
PTXBuilder::newOperand(mlir::Value value, StringRef constraint,
                       std::function<std::string(int)> formater) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formater;
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

PTXBuilder::Operand *PTXBuilder::newConstantOperand(int v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return newConstantOperand(ss.str());
}

std::string PTXBuilder::getConstrains() const {
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

SmallVector<PTXBuilder::Operand *> PTXBuilder::getAllArgs() const {
  llvm::SmallVector<Operand *, 4> res;
  for (auto &x : argArchive)
    if (!x->isList())
      res.push_back(x.get());
  return res;
}

std::string PTXInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return llvm::formatv("${0}", idx);

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return "{ " + strJoin(oprs, ", ") + " }";
}

PTXInstr::Operand *PTXBuilder::newAddrOperand(mlir::Value addr,
                                              StringRef constraint, int off) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [off](int idx) -> std::string {
    return llvm::formatv("[ ${0} + {1} ]", idx, off);
  };

  return opr;
}

std::string PTXBuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &instr : instrs) {
    lines.push_back(instr->dump());
  }

  return strJoin(lines, "\n\t");
}

std::string PTXInstrCommon::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);
  if (pred)
    if (!pred->repr)
      os << "@" << pred->dump() << " ";
    else
      os << pred->repr(pred->idx);

  std::string instrRepr = strJoin(instrParts, ".");

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = strJoin(argReprs, ", ");

  os << instrRepr << " " << argsRepr << ";";
  os.flush();
  return osStr;
}

SmallVector<PTXInstrCommon::Operand *> PTXInstrCommon::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

void PTXInstrCommon::operator()(ArrayRef<Operand *> oprs) {
  for (auto *opr : oprs) {
    addOperand(opr);
  }
}
} // namespace triton
} // namespace mlir
