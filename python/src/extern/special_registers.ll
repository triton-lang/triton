; globaltimer.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "nvptx64-nvidia-cuda"


define i64 @globaltimer() #0 {
  %1 = call i64 asm sideeffect "mov.u64 $0, %globaltimer;", "=l"() nounwind
  ret i64 %1
}

attributes #0 = { alwaysinline nounwind }
