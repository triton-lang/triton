#ifndef TRITONGPU_TRANSFORM_WSUTILITY_H_
#define TRITONGPU_TRANSFORM_WSUTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {

namespace gpu {

constexpr char ATTR_WS_MANUAL[] = "ttg.manual-nvws";
constexpr char ATTR_WS_PREFIX[] = "nvws.";
constexpr char ATTR_WS_TMALOAD[] = "nvws.tma_load";
constexpr char ATTR_WS_MMA[] = "nvws.mma";
constexpr char ATTR_WS_EPILOGUE[] = "nvws.epilogue";
constexpr char ATTR_WSGROUPS[] = "groups";
constexpr char ATTR_WS_AREF_IF[] = "aref_if";
constexpr char ATTR_WS_INIT_BARRIER_SYNC[] = "init_barrier_sync";
constexpr char ATTR_WS_BARID[] = "barId";

struct WSGroup {
  int startWarp;
  int numWarps;
};

std::set<std::string> getGroups(Operation *op);
std::set<std::string> getGroups(OpResult result);

void setGroups(Operation *op, const std::set<std::string> &groups);
void setGroups(OpResult result, const std::set<std::string> &groups);

void addGroups(Operation *op, const std::set<std::string> &groups);
void addGroups(OpResult result, const std::set<std::string> &groups);

void copyGroups(Operation *from_op, Operation *to_op);

class OpBuilderWithGroup : public OpBuilder {
public:
  explicit OpBuilderWithGroup(Operation *op, std::string wsGroup = {})
      : OpBuilder(op), group(wsGroup) {}

  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(location, std::forward<Args>(args)...);
    if (!group.empty()) {
      setGroups(op, {group});
    }
    return op;
  }
  using OpBuilder::create;

private:
  std::string group;
};

SmallVector<nvidia_gpu::WarpGroupOp> findWarpGroupOps(ModuleOp m);

void setGroupAttribute(ModuleOp moduleOp, const std::string &name,
                       WSGroup group);
void setGroupAttribute(ModuleOp moduleOp, const std::string &name,
                       int startWarp, int numWarps);
SymbolRefAttr mkGroup(ModuleOp moduleOp, const std::string &name,
                      WSGroup group);

std::string getGroup(nvidia_gpu::WarpGroupOp wgOp);
WSGroup getGroupFromSymbolRefAttr(ModuleOp mod, SymbolRefAttr refAttr);

bool isOpInGroup(Operation *op, const std::string &group);
bool isResultInGroup(Value value, const std::string &group);
Value getLoopNumIter(Value lb, Value ub, Value step, Location loc,
                     OpBuilder &builder);
Value getLoopNumIter(scf::ForOp forOp, OpBuilder &builder);
int getBarrierID(nvidia_gpu::WarpGroupOp wgOp);

bool isMMAOp(Operation *op);
bool isInnerMostLoop(scf::ForOp forOp);
bool isFMHAMathLoop(scf::ForOp forOp);

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITONGPU_TRANSFORM_WSUTILITY_H_
