#include "triton/Conversion/TritonGPUToLLVM/PtxAsmFormat.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {

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

std::string PtxInstr::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);
  if (pred)
    os << "@" << pred->dump() << " ";

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

llvm::SmallVector<PtxInstr::Operand *, 4> PtxInstr::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

PtxInstr::Operand *
PtxInstr::newOperand(mlir::Value value, StringRef constraint,
                     std::function<std::string(int)> formater) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formater;
  opr->idx = oprCounter++;
  return opr;
}

std::string PtxInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return llvm::formatv("${0}", idx);
  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return "{ " + strJoin(oprs, ", ") + " }";
}

PtxInstr::Operand *PtxIOInstr::newAddrOperand(mlir::Value addr,
                                              StringRef constraint, int off) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [off](int idx) -> std::string {
    return llvm::formatv("[ ${0} + {1} ]", idx, off);
  };

  return opr;
}
} // namespace triton
} // namespace mlir
