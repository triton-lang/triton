; RUN: triton-llvm-opt -break-struct-phi-nodes %s | FileCheck %s
; RUN: triton-llvm-opt -merge-nvptx-assert-blocks %s | FileCheck %s --check-prefix=ASSERT

; CHECK-LABEL: struct
define {i32, i32} @struct(i1 %c) {
; CHECK: br i1 %{{.*}}, label [[TRUE:%.*]], label [[FALSE:%.*]]
  br i1 %c, label %true, label %false

true:
  %s.1 = insertvalue {i32, i32} undef, i32 20, 0
  %s.2 = insertvalue {i32, i32} %s.1, i32 200, 1

; CHECK-DAG: [[E0:%.*]] = extractvalue { i32, i32 } %{{.*}}, 0
; CHECK-DAG: [[E1:%.*]] = extractvalue { i32, i32 } %{{.*}}, 1
; CHECK: br
  br label %exit

false:
  %s.3 = insertvalue {i32, i32} undef, i32 30, 0
  %s.4 = insertvalue {i32, i32} %s.3, i32 300, 1
; CHECK-DAG: [[E2:%.*]] = extractvalue { i32, i32 } %{{.*}}, 0
; CHECK-DAG: [[E3:%.*]] = extractvalue { i32, i32 } %{{.*}}, 1
; CHECK: br
  br label %exit

exit:
; CHECK-DAG: [[PHI0:%.*]] = phi i32 [ [[E0]], [[TRUE]] ], [ [[E2]], [[FALSE]] ]
; CHECK-DAG: [[PHI1:%.*]] = phi i32 [ [[E1]], [[TRUE]] ], [ [[E3]], [[FALSE]] ]
; CHECK: [[S0:%.*]] = insertvalue { i32, i32 } undef, i32 [[PHI0]], 0
; CHECK: [[S1:%.*]] = insertvalue { i32, i32 } [[S0]], i32 [[PHI1]], 1
; CHECK: ret { i32, i32 } [[S1]]
  %r = phi {i32, i32} [ %s.2, %true], [ %s.4, %false ]
  ret {i32, i32} %r
}

; Preserve distinct source diagnostics and observable failure-side effects.
declare void @__assertfail(ptr, ptr, i32, ptr, i64) noreturn
declare void @observe()

; ASSERT-LABEL: define void @checks(
; ASSERT: br i1 %a, label %failure, label %second
; ASSERT: br i1 %b, label %failure, label %third
; ASSERT: br i1 %c, label %different_line, label %observable
; ASSERT: failure:
; ASSERT-NEXT: call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1)
; ASSERT-NEXT: unreachable
; ASSERT-NOT: repeated:
; ASSERT: different_line:
; ASSERT-NEXT: call void @__assertfail(ptr %msg, ptr %file, i32 13, ptr %fn, i64 1)
; ASSERT: observable:
; ASSERT-NEXT: call void @observe()
; ASSERT-NEXT: call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1)
define void @checks(i1 %a, i1 %b, i1 %c, ptr %msg, ptr %file, ptr %fn) {
entry:
  br i1 %a, label %failure, label %second
second:
  br i1 %b, label %repeated, label %third
third:
  br i1 %c, label %different_line, label %observable
failure:
  call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1)
  unreachable
repeated:
  call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1)
  unreachable
different_line:
  call void @__assertfail(ptr %msg, ptr %file, i32 13, ptr %fn, i64 1)
  unreachable
observable:
  call void @observe()
  call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1)
  unreachable
}

; The same diagnostic operands can occur at independently debuggable call sites.
; ASSERT-LABEL: define void @different_debug_locations(
; ASSERT: br i1 %cond, label %first, label %second
; ASSERT: first:
; ASSERT: call void @__assertfail
; ASSERT: second:
; ASSERT: call void @__assertfail
define void @different_debug_locations(i1 %cond, ptr %msg, ptr %file, ptr %fn) !dbg !4 {
entry:
  br i1 %cond, label %first, label %second
first:
  call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1), !dbg !6
  unreachable
second:
  call void @__assertfail(ptr %msg, ptr %file, i32 12, ptr %fn, i64 1), !dbg !7
  unreachable
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "kernel.py", directory: "/")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "different_debug_locations", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DILocation(line: 20, column: 2, scope: !4)
!7 = !DILocation(line: 21, column: 2, scope: !4)
