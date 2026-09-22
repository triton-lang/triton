#include "Dump/TreeDataDump.h"

#include <ostream>

namespace proton {

void TreeData::dumpHatchet(std::ostream &os, size_t phase) const {
  treePhases.withPtr(phase, [&](Tree *tree) {
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      auto output = buildHatchetJson(tree, virtualTree);
      os << std::endl << output.dump(4) << std::endl;
    });
  });
}

void TreeData::dumpHatchetMsgPack(std::ostream &os, size_t phase) const {
  treePhases.withPtr(phase, [&](Tree *tree) {
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      auto msgPack = buildHatchetMsgPack(tree, virtualTree);
      os.write(reinterpret_cast<const char *>(msgPack.data()),
               static_cast<std::streamsize>(msgPack.size()));
    });
  });
}

std::string TreeData::toJsonString(size_t phase) const {
  return treePhases.withPtr(phase, [&](Tree *tree) {
    return treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      return buildHatchetJson(tree, virtualTree).dump();
    });
  });
}

std::vector<uint8_t> TreeData::toMsgPack(size_t phase) const {
  return treePhases.withPtr(phase, [&](Tree *tree) {
    return treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      return buildHatchetMsgPack(tree, virtualTree);
    });
  });
}

} // namespace proton
