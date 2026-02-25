#include <llvm>
#include <unordered_map>
#include <deque>

namespace mlir{
	
void NoteAnalysis::run(FuncBlockMemberMapT &funcBlockMemberMap)  {
	auto funcOp = dyn_cast <FunctionOpInterface>(allocation->getOperation());
	if (!funcOp) return; // Handles member notes and cases where the cast fails
	OpBuilder builder(funcOp.getContext());
	resolve(funcOp, &funcBlockMemberMap, &builder);
}
