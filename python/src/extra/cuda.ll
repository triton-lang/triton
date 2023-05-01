; ~/.triton/llvm/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release/bin/llvm-as ./src/extra/cuda.ll -o ./triton/language/extra/cuda.bc

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "nvptx64-nvidia-cuda"


define i64 @globaltimer() #0 {
  %1 = call i64 asm sideeffect "mov.u64 $0, %globaltimer;", "=l"() nounwind
  ret i64 %1
}

define i32 @smid() #0 {
  %1 = call i32 asm "mov.u32 $0, %smid;", "=r"() nounwind
  ret i32 %1
}

attributes #0 = { alwaysinline nounwind }
