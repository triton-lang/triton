# SYMBOL=_ZN4llvm11PassBuilder17OptimizationLevel2O0E
# SYMBOL=_ZN4llvm11DDGAnalysis3KeyE
# SYMBOL=_ZN4llvm26UnifyFunctionExitNodesPass3runERNS_8FunctionERNS_15AnalysisManagerIS1_JEEE
# SYMBOL=_ZN4llvm12LoopFusePass3runERNS_8FunctionERNS_15AnalysisManagerIS1_JEEE
# SYMBOL=_ZN4llvm30moveInstructionsToTheBeginningERNS_10BasicBlockES1_RNS_13DominatorTreeERKNS_17PostDominatorTreeERNS_14DependenceInfoE
# SYMBOL=_ZN4llvm17LoopExtractorPass3runERNS_6ModuleERNS_15AnalysisManagerIS1_JEEE
# SYMBOL=_ZN4llvm17ObjCARCExpandPass3runERNS_8FunctionERNS_15AnalysisManagerIS1_JEEE
# SYMBOL=_ZN4llvm13CoroSplitPass3runERNS_13LazyCallGraph3SCCERNS_15AnalysisManagerIS2_JRS1_EEES5_RNS_17CGSCCUpdateResultE
SYMBOL=_ZN4llvm20SyntheticCountsUtilsIPKNS_9CallGraphEE9propagateERKS3_NS_12function_refIFNS_8OptionalINS_12ScaledNumberImEEEEPKNS_13CallGraphNodeERKSt4pairINS8_INS_14WeakTrackingVHEEEPSC_EEEENS7_IFvSE_SA_EEE
for lib in $(find /tmp/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04/ -name \*.a); do
    symbols=$(nm $lib | grep $SYMBOL | grep -v " U ")
    
    if [ "${#symbols}" -gt "0" ]; then
        echo $lib
        echo $symbols
    fi

done
