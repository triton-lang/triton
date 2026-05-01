; ModuleID = '/Users/jeffniu/code/triton/python/triton/experimental/gsan/src/GSanLibrary.cu'
source_filename = "/Users/jeffniu/code/triton/python/triton/experimental/gsan/src/GSanLibrary.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"struct.gsan::Location" = type { ptr, i32 }
%"struct.gsan::AtomicEventState" = type { ptr, [3 x ptr], i8 }

@.str = private unnamed_addr constant [10 x i8] c"<unknown>\00", align 1
@.str1 = private unnamed_addr constant [31 x i8] c"Read after write race detected\00", align 1
@.str2 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str3 = private unnamed_addr constant [25 x i8] c"Invalid GSan clock token\00", align 1
@.str4 = private unnamed_addr constant [24 x i8] c"Future GSan clock token\00", align 1
@.str5 = private unnamed_addr constant [36 x i8] c"GSan clock buffer token overwritten\00", align 1
@.str6 = private unnamed_addr constant [24 x i8] c"Vector clock overflowed\00", align 1
@.str7 = private unnamed_addr constant [31 x i8] c"Write after read race detected\00", align 1
@.str8 = private unnamed_addr constant [32 x i8] c"Write after write race detected\00", align 1
@.str9 = private unnamed_addr constant [47 x i8] c"Atomic access spans too many GSan shadow cells\00", align 1
@.str10 = private unnamed_addr constant [40 x i8] c"GSan clock buffer size must be non-zero\00", align 1
@.str11 = private unnamed_addr constant [35 x i8] c"GSan clock buffer token overflowed\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn denormal_fpenv(float: preservesign) memory(argmem: read)
define dso_local noundef nonnull ptr @_ZN4gsan13getSourceFileENS_8LocationE(ptr noundef readonly byval(%"struct.gsan::Location") align 8 captures(none) %loc) local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr %loc, align 8, !tbaa !7
  %cmp = icmp eq ptr %0, null
  %cond = select i1 %cmp, ptr @.str, ptr %0
  ret ptr %cond
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_load_tensor(ptr noundef %globalState, ptr noundef readonly captures(none) %stackPtr, i32 noundef %numElems, i32 noundef %bytesPerElem, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i.i = add i64 %1, 39
  %cond.i.i.i = and i64 %ptr.biased.i.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val22.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i = zext i16 %globals.val22.i to i64
  %add.i.i = add nuw nsw i64 %conv.i.i, 1
  %conv1.i.i = zext i16 %globals.val.i to i64
  %mul.i.i = shl nuw nsw i64 %conv1.i.i, 1
  %mul3.i.i = mul nuw nsw i64 %mul.i.i, %add.i.i
  %add4.i.i = add nuw nsw i64 %mul3.i.i, 32
  %conv.i = zext i32 %0 to i64
  %mul.i = mul i64 %add4.i.i, %conv.i
  %add.i = add i64 %mul.i, %cond.i.i.i
  %4 = inttoptr i64 %add.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i = icmp eq ptr %5, null
  br i1 %cmp.i, label %if.then.i, label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

if.then.i:                                        ; preds = %entry
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase4.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase4.i, align 8, !tbaa !19
  %numReads.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i, align 8, !tbaa !3
  %clockBufferDirty.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i, align 4
  %globalsBase1.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i, align 8, !tbaa !20
  %sub.i.i = sub i64 %1, %7
  %div6.i.i = lshr i64 %sub.i.i, 30
  %numSms.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i, align 4, !tbaa !21
  %conv.i23.i = zext i16 %8 to i64
  %mul.i24.i = mul nuw nsw i64 %div6.i.i, %conv.i23.i
  %add.i25.i = add nuw nsw i64 %mul.i24.i, %conv.i
  %conv3.i.i = trunc i64 %add.i25.i to i16
  %threadId.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i, ptr %threadId.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit: ; preds = %entry, %if.then.i
  %conv.i4 = sext i32 %numElems to i64
  %mul.i5 = shl nsw i64 %conv.i4, 3
  %add.ptr.i = getelementptr inbounds nuw i8, ptr %stackPtr, i64 %mul.i5
  %cmp9.i = icmp sgt i32 %numElems, 0
  br i1 %cmp9.i, label %for.body.lr.ph.i, label %_ZN4gsan12_GLOBAL__N_110tensorLoadEPNS_11ThreadStateEPKciiNS_8LocationE.exit

for.body.lr.ph.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  %conv.i.i6 = sext i32 %bytesPerElem to i64
  %reserveBase1.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  %lock.i.i = getelementptr inbounds nuw i8, ptr %4, i64 24
  %and.i.i.i.i.i.i.i = and i64 %add.i, -1073741824
  %9 = inttoptr i64 %and.i.i.i.i.i.i.i to ptr
  %numSms.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %9, i64 20
  %globalsBase.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %9, i64 8
  %10 = getelementptr i8, ptr %9, i64 24
  %11 = getelementptr i8, ptr %9, i64 26
  %vectorClock.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %threadId.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  %numReads18.i.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  %rngSeed.i.i.i.i = getelementptr inbounds nuw i8, ptr %9, i64 16
  %cmp.i37.i.i.i.i.i.i.i = icmp eq ptr %file, null
  %cond.i38.i.i.i.i.i.i.i = select i1 %cmp.i37.i.i.i.i.i.i.i, ptr @.str, ptr %file
  br label %for.body.i

for.body.i:                                       ; preds = %if.end.i, %for.body.lr.ph.i
  %i.010.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %if.end.i ]
  %idxprom.i = zext nneg i32 %i.010.i to i64
  %arrayidx2.i = getelementptr inbounds nuw i8, ptr %add.ptr.i, i64 %idxprom.i
  %12 = load i8, ptr %arrayidx2.i, align 1, !tbaa !23
  %tobool.not.i = icmp eq i8 %12, 0
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i7

if.then.i7:                                       ; preds = %for.body.i
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %stackPtr, i64 %idxprom.i
  %13 = load i64, ptr %arrayidx.i, align 8, !tbaa !19
  %add.i.i8 = add i64 %13, %conv.i.i6
  %sub.i.i.i = and i64 %13, -4
  %rem3.i.i.i = and i64 %add.i.i8, 3
  %cmp.i.i.i = icmp eq i64 %rem3.i.i.i, 0
  %sub5.i.i.i = sub nuw nsw i64 4, %rem3.i.i.i
  %cond.i.i.i9 = select i1 %cmp.i.i.i, i64 0, i64 %sub5.i.i.i
  %add.i.i.i = add i64 %cond.i.i.i9, %add.i.i8
  %14 = load i64, ptr %reserveBase1.i.i, align 8, !tbaa !19
  %15 = atomicrmw add ptr %lock.i.i, i32 1 syncscope("block") acquire, align 4
  %cmp.i19.i.i = icmp sgt i32 %15, -1
  br i1 %cmp.i19.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i

do.body.i.i.i:                                    ; preds = %if.then.i7, %do.body.i.i.i
  %16 = load atomic i32, ptr %lock.i.i syncscope("block") acquire, align 4
  %cmp3.not.i.i.i = icmp sgt i32 %16, -1
  br i1 %cmp3.not.i.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i, !llvm.loop !24

_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i: ; preds = %do.body.i.i.i, %if.then.i7
  %cmp26.i.i = icmp ult i64 %sub.i.i.i, %add.i.i.i
  br i1 %cmp26.i.i, label %for.body.i.i.preheader, label %_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i

for.body.i.i.preheader:                           ; preds = %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %invariant.op = sub i64 -549755813888, %14
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.preheader, %for.inc.i.i
  %addr.027.i.i = phi i64 [ %add8.i.i, %for.inc.i.i ], [ %sub.i.i.i, %for.body.i.i.preheader ]
  %and.i.i.i.i = and i64 %addr.027.i.i, -1099511627776
  %cmp.i20.i.i = icmp eq i64 %and.i.i.i.i, %14
  br i1 %cmp.i20.i.i, label %if.end.i.i, label %for.inc.i.i

if.end.i.i:                                       ; preds = %for.body.i.i
  %sub.i22.reass.i.reass.i.reass.reass = add i64 %addr.027.i.i, %invariant.op
  %div4.i.i.i = lshr exact i64 %sub.i22.reass.i.reass.i.reass.reass, 2
  %mul.i.i.i = mul i64 %div4.i.i.i, 24
  %add.i23.i.i = add i64 %mul.i.i.i, %14
  %17 = inttoptr i64 %add.i23.i.i to ptr
  %lock.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 22
  br label %while.cond.i.i.i

while.cond.i.i.i:                                 ; preds = %while.cond.i.i.i, %if.end.i.i
  %18 = cmpxchg weak ptr %lock.i.i.i, i16 0, i16 1 acquire monotonic, align 2
  %19 = extractvalue { i16, i1 } %18, 1
  br i1 %19, label %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, label %while.cond.i.i.i, !llvm.loop !26

_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i: ; preds = %while.cond.i.i.i
  %writeClock.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 16
  %20 = load i16, ptr %writeClock.i.i.i, align 4, !tbaa !27
  %cmp.i.i.i.i = icmp eq i16 %20, 0
  br i1 %cmp.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i.i.i, label %if.end.i.i.i.i

if.end.i.i.i.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i
  %.phi.trans.insert.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 18
  %clock.val14.i.pre.i.i.i.i = load i16, ptr %.phi.trans.insert.i.i.i.i, align 2
  %21 = and i16 %clock.val14.i.pre.i.i.i.i, 16384
  %bf.cast.not.i.i.i.i.i.i = icmp eq i16 %21, 0
  br i1 %bf.cast.not.i.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i, label %if.end.i.i.i.i.i.i

if.end.i.i.i.i.i.i:                               ; preds = %if.end.i.i.i.i
  %bf.clear2.i.i.i.i.i.i = and i16 %clock.val14.i.pre.i.i.i.i, 4095
  %22 = load i16, ptr %numSms.i.i.i.i.i.i.i, align 4, !tbaa !21
  %bf.clear2.i.i.i.i.i.i.frozen = freeze i16 %bf.clear2.i.i.i.i.i.i
  %.frozen = freeze i16 %22
  %div16.i.i.i.i.i.i.i = udiv i16 %bf.clear2.i.i.i.i.i.i.frozen, %.frozen
  %23 = mul i16 %div16.i.i.i.i.i.i.i, %.frozen
  %rem17.i.i.i.i.i.i.i.decomposed = sub i16 %bf.clear2.i.i.i.i.i.i.frozen, %23
  %24 = load i64, ptr %globalsBase.i.i.i.i.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i.i.i.i = zext nneg i16 %div16.i.i.i.i.i.i.i to i64
  %mul.i.i.i.i.i.i.i = shl nuw nsw i64 %conv5.i.i.i.i.i.i.i, 30
  %add.i.i.i.i.i.i.i = or disjoint i64 %mul.i.i.i.i.i.i.i, 39
  %ptr.biased.i.i.i.i.i.i.i.i.i = add i64 %add.i.i.i.i.i.i.i, %24
  %cond.i.i.i.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i.i.i.i = load i16, ptr %10, align 8, !tbaa !11
  %globals.val15.i.i.i.i.i.i.i = load i16, ptr %11, align 2, !tbaa !15
  %conv.i.i.i.i.i.i.i.i = zext i16 %globals.val15.i.i.i.i.i.i.i to i64
  %add.i.i.i.i.i.i.i.i = add nuw nsw i64 %conv.i.i.i.i.i.i.i.i, 1
  %conv1.i.i.i.i.i.i.i.i = zext i16 %globals.val.i.i.i.i.i.i.i to i64
  %mul.i.i.i.i.i.i.i.i = shl nuw nsw i64 %conv1.i.i.i.i.i.i.i.i, 1
  %mul3.i.i.i.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i.i.i.i.i, %add.i.i.i.i.i.i.i.i
  %add4.i.i.i.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i.i.i.i, 32
  %conv7.i.i.i.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i.i, %conv7.i.i.i.i.i.i.i
  %add9.i.i.i.i.i.i.i = add i64 %mul8.i.i.i.i.i.i.i, %cond.i.i.i.i.i.i.i.i.i
  %25 = inttoptr i64 %add9.i.i.i.i.i.i.i to ptr
  %conv.i.i.i.i.i.i.i = zext i16 %20 to i32
  %clockBufferHead.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %25, i64 20
  %bf.load.i.i.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i.i.i, align 4
  %bf.lshr.i.i.i.i.i.i.i = lshr i32 %bf.load.i.i.i.i.i.i.i, 1
  %cmp3.not.i.i.i.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i
  br i1 %cmp3.not.i.i.i.i.i.i.i, label %if.then4.i.i.i.i.i.i.i, label %do.end9.i.i.i.i.i.i.i

if.then4.i.i.i.i.i.i.i:                           ; preds = %if.end.i.i.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i.i.i, align 4
  %.pre42.i.i.i.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i.i.i.i, 1
  br label %do.end9.i.i.i.i.i.i.i

do.end9.i.i.i.i.i.i.i:                            ; preds = %if.then4.i.i.i.i.i.i.i, %if.end.i.i.i.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i.i.i.i, %if.end.i.i.i.i.i.i ], [ %.pre42.i.i.i.i.i.i.i, %if.then4.i.i.i.i.i.i.i ]
  %and.i.i.i.i.i.i.i.i = and i64 %add9.i.i.i.i.i.i.i, -1073741824
  %26 = inttoptr i64 %and.i.i.i.i.i.i.i.i to ptr
  %sub.i.i.i.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i
  %clockBufferSize.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %26, i64 26
  %27 = load i16, ptr %clockBufferSize.i.i.i.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i.i.i.i = zext i16 %27 to i32
  %cmp17.i.i.i.i.i.i.i = icmp slt i32 %sub.i.i.i.i.i.i.i, %conv16.i.i.i.i.i.i.i
  br i1 %cmp17.i.i.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i, label %if.then18.i.i.i.i.i.i.i

if.then18.i.i.i.i.i.i.i:                          ; preds = %do.end9.i.i.i.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i38.i.i.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i: ; preds = %if.then18.i.i.i.i.i.i.i, %do.end9.i.i.i.i.i.i.i
  %28 = phi i16 [ %.pre.i.i.i.i.i.i.i, %if.then18.i.i.i.i.i.i.i ], [ %27, %do.end9.i.i.i.i.i.i.i ]
  %29 = urem i16 %20, %28
  %rem.i.i.i.i.i.i.i = zext i16 %29 to i64
  %vectorClock.i.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %25, i64 30
  %numThreads.i.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %26, i64 24
  %30 = load i16, ptr %numThreads.i.i.i.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i.i.i.i = zext i16 %30 to i64
  %add.ptr.i.i.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i.i.i.i
  %mul.i8.i.i.i.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i.i.i.i, %rem.i.i.i.i.i.i.i
  %add.ptr.i.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i.i.i.i, i64 %mul.i8.i.i.i.i.i.i
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i, %if.end.i.i.i.i
  %retval.0.i.i22.i.i.i.i = phi ptr [ %add.ptr.i.i.i.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i ], [ null, %if.end.i.i.i.i ]
  %tobool.not.not.i.i.i.i.i = icmp eq ptr %retval.0.i.i22.i.i.i.i, null
  br i1 %tobool.not.not.i.i.i.i.i, label %cleanup.i.i.i.i.i, label %if.then1.i.i.i.i.i

if.then1.i.i.i.i.i:                               ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i
  %31 = load i16, ptr %10, align 8, !tbaa !11
  %conv.i.i.i.i.i.i = zext i16 %31 to i32
  %cmp.not11.i.i.i.i.i.i = icmp eq i16 %31, 0
  br i1 %cmp.not11.i.i.i.i.i.i, label %cleanup.i.i.i.i.i, label %for.body.i.i.i.i.i.i

for.body.i.i.i.i.i.i:                             ; preds = %if.then1.i.i.i.i.i, %for.body.i.i.i.i.i.i
  %i.012.i.i.i.i.i.i = phi i32 [ %inc.i.i.i.i.i.i, %for.body.i.i.i.i.i.i ], [ 0, %if.then1.i.i.i.i.i ]
  %idxprom.i.i.i.i.i.i = zext nneg i32 %i.012.i.i.i.i.i.i to i64
  %arrayidx.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i, i64 %idxprom.i.i.i.i.i.i
  %32 = load i16, ptr %arrayidx.i.i.i.i.i.i, align 2, !tbaa !22
  %arrayidx3.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i.i.i.i, i64 %idxprom.i.i.i.i.i.i
  %33 = load i16, ptr %arrayidx3.i.i.i.i.i.i, align 2, !tbaa !22
  %cmp5.i.i.i.i.i.i = icmp uge i16 %32, %33
  %inc.i.i.i.i.i.i = add nuw nsw i32 %i.012.i.i.i.i.i.i, 1
  %exitcond.not.i.i.i.i.i.i = icmp ne i32 %inc.i.i.i.i.i.i, %conv.i.i.i.i.i.i
  %or.cond.not.i.i.i.i.i.i = select i1 %cmp5.i.i.i.i.i.i, i1 %exitcond.not.i.i.i.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i, label %cleanup.i.i.i.i.i, !llvm.loop !31

cleanup.i.i.i.i.i:                                ; preds = %for.body.i.i.i.i.i.i, %if.then1.i.i.i.i.i, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i
  %retval.0.i23.i.i.i.i = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i ], [ true, %if.then1.i.i.i.i.i ], [ %cmp5.i.i.i.i.i.i, %for.body.i.i.i.i.i.i ]
  br i1 %tobool.not.not.i.i.i.i.i, label %cleanup.cont.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i

cleanup.cont.i.i.i.i.i:                           ; preds = %cleanup.i.i.i.i.i
  %bf.load.i.i.i.i.i = load i16, ptr %.phi.trans.insert.i.i.i.i, align 2
  %bf.clear.i.i.i.i.i = and i16 %bf.load.i.i.i.i.i, 4095
  %idxprom.i.i.i.i.i = zext nneg i16 %bf.clear.i.i.i.i.i to i64
  %arrayidx.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i, i64 %idxprom.i.i.i.i.i
  %34 = load i16, ptr %arrayidx.i.i.i.i.i, align 2, !tbaa !22
  %35 = load i16, ptr %writeClock.i.i.i, align 4, !tbaa !27
  %cmp7.i.i.i.i.i = icmp uge i16 %34, %35
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i: ; preds = %cleanup.cont.i.i.i.i.i, %cleanup.i.i.i.i.i
  %retval.1.i.i.i.i.i = phi i1 [ %retval.0.i23.i.i.i.i, %cleanup.i.i.i.i.i ], [ %cmp7.i.i.i.i.i, %cleanup.cont.i.i.i.i.i ]
  br i1 %retval.1.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i.i.i, label %if.then9.i.i.i.i

if.then9.i.i.i.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str1, ptr noundef nonnull %cond.i38.i.i.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i.i.i

_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i.i.i: ; preds = %if.then9.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i, %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i
  %numReads1.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 20
  %36 = load i16, ptr %numReads1.i.i.i.i, align 4, !tbaa !32
  %conv.i.i.i.i = zext i16 %36 to i32
  %cmp.not.i.i.i.i = icmp eq i16 %36, -1
  br i1 %cmp.not.i.i.i.i, label %if.end.i4.i.i.i, label %if.then.i.i.i.i

if.then.i.i.i.i:                                  ; preds = %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i.i.i
  %inc.i.i.i.i = add nuw i16 %36, 1
  store i16 %inc.i.i.i.i, ptr %numReads1.i.i.i.i, align 4, !tbaa !32
  br label %if.end.i4.i.i.i

if.end.i4.i.i.i:                                  ; preds = %if.then.i.i.i.i, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i.i.i
  %37 = load i16, ptr %threadId.i.i.i.i.i, align 4, !tbaa !22
  %idxprom.i.i6.i.i.i = zext i16 %37 to i64
  %arrayidx.i.i7.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i, i64 %idxprom.i.i6.i.i.i
  %38 = load i16, ptr %arrayidx.i.i7.i.i.i, align 2, !tbaa !22
  %bf.value.i.i.i.i.i = and i16 %37, 4095
  %readClock.sroa.0.0.copyload.i.i.i.i = load i16, ptr %17, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 2
  %readClock.sroa.4.0.copyload.i.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.i.i.i.i, align 2, !tbaa !23
  %bf.clear.i.i.i.i = and i16 %readClock.sroa.4.0.copyload.i.i.i.i, 4095
  %cmp7.i.i.i.i = icmp ne i16 %bf.clear.i.i.i.i, %37
  %cmp9.i.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.i.i.i.i, 0
  %or.cond.not.i.i.i.i = select i1 %cmp7.i.i.i.i, i1 %cmp9.i.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i.i, label %for.cond.i.i.i.i, label %if.then10.i.i.i.i

for.cond.i.i.i.i:                                 ; preds = %if.end.i4.i.i.i
  %arrayidx.1.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 4
  %readClock.sroa.0.0.copyload.1.i.i.i.i = load i16, ptr %arrayidx.1.i.i.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.1.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 6
  %readClock.sroa.4.0.copyload.1.i.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.1.i.i.i.i, align 2, !tbaa !23
  %bf.clear.1.i.i.i.i = and i16 %readClock.sroa.4.0.copyload.1.i.i.i.i, 4095
  %cmp7.1.i.i.i.i = icmp ne i16 %bf.clear.1.i.i.i.i, %37
  %cmp9.1.i.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.1.i.i.i.i, 0
  %or.cond.not.1.i.i.i.i = select i1 %cmp7.1.i.i.i.i, i1 %cmp9.1.i.i.i.i, i1 false
  br i1 %or.cond.not.1.i.i.i.i, label %for.cond.1.i.i.i.i, label %if.then10.i.i.i.i

for.cond.1.i.i.i.i:                               ; preds = %for.cond.i.i.i.i
  %arrayidx.2.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 8
  %readClock.sroa.0.0.copyload.2.i.i.i.i = load i16, ptr %arrayidx.2.i.i.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.2.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 10
  %readClock.sroa.4.0.copyload.2.i.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.2.i.i.i.i, align 2, !tbaa !23
  %bf.clear.2.i.i.i.i = and i16 %readClock.sroa.4.0.copyload.2.i.i.i.i, 4095
  %cmp7.2.i.i.i.i = icmp ne i16 %bf.clear.2.i.i.i.i, %37
  %cmp9.2.i.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.2.i.i.i.i, 0
  %or.cond.not.2.i.i.i.i = select i1 %cmp7.2.i.i.i.i, i1 %cmp9.2.i.i.i.i, i1 false
  br i1 %or.cond.not.2.i.i.i.i, label %for.cond.2.i.i.i.i, label %if.then10.i.i.i.i

for.cond.2.i.i.i.i:                               ; preds = %for.cond.1.i.i.i.i
  %arrayidx.3.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 12
  %readClock.sroa.0.0.copyload.3.i.i.i.i = load i16, ptr %arrayidx.3.i.i.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.3.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 14
  %readClock.sroa.4.0.copyload.3.i.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.3.i.i.i.i, align 2, !tbaa !23
  %bf.clear.3.i.i.i.i = and i16 %readClock.sroa.4.0.copyload.3.i.i.i.i, 4095
  %cmp7.3.i.i.i.i = icmp ne i16 %bf.clear.3.i.i.i.i, %37
  %cmp9.3.i.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.3.i.i.i.i, 0
  %or.cond.not.3.i.i.i.i = select i1 %cmp7.3.i.i.i.i, i1 %cmp9.3.i.i.i.i, i1 false
  br i1 %or.cond.not.3.i.i.i.i, label %for.cond.3.i.i.i.i, label %if.then10.i.i.i.i

for.cond.3.i.i.i.i:                               ; preds = %for.cond.2.i.i.i.i
  %39 = atomicrmw add ptr %numReads18.i.i.i.i, i32 1 syncscope("block") monotonic, align 8
  %40 = load i32, ptr %rngSeed.i.i.i.i, align 16, !tbaa !34
  %conv21.i.i.i.i = zext i16 %37 to i32
  %mul.i.i.i.i8.i.i.i = mul i32 %39, -862048943
  %or.i.i.i.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i.i.i8.i.i.i, i32 %mul.i.i.i.i8.i.i.i, i32 15)
  %mul1.i.i.i.i.i.i.i = mul i32 %or.i.i.i.i.i.i.i.i, 461845907
  %xor.i.i.i.i.i.i = xor i32 %mul1.i.i.i.i.i.i.i, %40
  %or.i.i.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i.i.i.i.i.i, i32 %xor.i.i.i.i.i.i, i32 13)
  %mul.i.i.i.i.i.i = mul i32 %or.i.i.i.i.i.i.i, 5
  %add.i.i.i.i.i.i = add i32 %mul.i.i.i.i.i.i, -430675100
  %mul.i.i6.i.i.i.i.i = mul i32 %conv21.i.i.i.i, -862048943
  %or.i.i.i7.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i6.i.i.i.i.i, i32 %mul.i.i6.i.i.i.i.i, i32 15)
  %mul1.i.i8.i.i.i.i.i = mul i32 %or.i.i.i7.i.i.i.i.i, 461845907
  %xor.i9.i.i.i.i.i = xor i32 %add.i.i.i.i.i.i, %mul1.i.i8.i.i.i.i.i
  %or.i.i10.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i9.i.i.i.i.i, i32 %xor.i9.i.i.i.i.i, i32 13)
  %mul.i11.i.i.i.i.i = mul i32 %or.i.i10.i.i.i.i.i, 5
  %add.i12.i.i.i.i.i = add i32 %mul.i11.i.i.i.i.i, -430675100
  %shr.i.i.i.i.i.i = lshr i32 %add.i12.i.i.i.i.i, 16
  %41 = xor i32 %add.i12.i.i.i.i.i, %shr.i.i.i.i.i.i
  %xor.i13.i.i.i.i.i = xor i32 %41, 8
  %mul.i14.i.i.i.i.i = mul i32 %xor.i13.i.i.i.i.i, -2048144789
  %shr1.i.i.i.i.i.i = lshr i32 %mul.i14.i.i.i.i.i, 13
  %xor2.i.i.i.i.i.i = xor i32 %shr1.i.i.i.i.i.i, %mul.i14.i.i.i.i.i
  %mul3.i.i.i.i.i.i = mul i32 %xor2.i.i.i.i.i.i, -1028477387
  %shr4.i.i.i.i.i.i = lshr i32 %mul3.i.i.i.i.i.i, 16
  %xor5.i.i.i.i.i.i = xor i32 %shr4.i.i.i.i.i.i, %mul3.i.i.i.i.i.i
  %rem.i.i.i.i = urem i32 %xor5.i.i.i.i.i.i, %conv.i.i.i.i
  %cmp24.i.i.i.i = icmp samesign ult i32 %rem.i.i.i.i, 4
  br i1 %cmp24.i.i.i.i, label %if.then25.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

if.then10.i.i.i.i:                                ; preds = %for.cond.2.i.i.i.i, %for.cond.1.i.i.i.i, %for.cond.i.i.i.i, %if.end.i4.i.i.i
  %arrayidx.lcssa.i.i.i.i = phi ptr [ %17, %if.end.i4.i.i.i ], [ %arrayidx.1.i.i.i.i, %for.cond.i.i.i.i ], [ %arrayidx.2.i.i.i.i, %for.cond.1.i.i.i.i ], [ %arrayidx.3.i.i.i.i, %for.cond.2.i.i.i.i ]
  %readClock.sroa.4.0.arrayidx.sroa_idx.le.i.i.i.i = getelementptr inbounds nuw i8, ptr %arrayidx.lcssa.i.i.i.i, i64 2
  store i16 %38, ptr %arrayidx.lcssa.i.i.i.i, align 4, !tbaa !22
  store i16 %bf.value.i.i.i.i.i, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.le.i.i.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

if.then25.i.i.i.i:                                ; preds = %for.cond.3.i.i.i.i
  %idxprom27.i.i.i.i = zext nneg i32 %rem.i.i.i.i to i64
  %arrayidx28.i.i.i.i = getelementptr inbounds nuw [4 x i8], ptr %17, i64 %idxprom27.i.i.i.i
  %scalarClock.sroa.5.0.arrayidx28.sroa_idx.i.i.i.i = getelementptr inbounds nuw i8, ptr %arrayidx28.i.i.i.i, i64 2
  store i16 %38, ptr %arrayidx28.i.i.i.i, align 4, !tbaa !22
  store i16 %bf.value.i.i.i.i.i, ptr %scalarClock.sroa.5.0.arrayidx28.sroa_idx.i.i.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i: ; preds = %if.then25.i.i.i.i, %if.then10.i.i.i.i, %for.cond.3.i.i.i.i
  store atomic i16 0, ptr %lock.i.i.i release, align 2
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i, %for.body.i.i
  %add8.i.i = add i64 %addr.027.i.i, 4
  %cmp.i.i = icmp ult i64 %add8.i.i, %add.i.i.i
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, !llvm.loop !35

_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i: ; preds = %for.inc.i.i, %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %42 = atomicrmw sub ptr %lock.i.i, i32 1 syncscope("block") monotonic, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, %for.body.i
  %inc.i = add nuw nsw i32 %i.010.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, %numElems
  br i1 %exitcond.not.i, label %_ZN4gsan12_GLOBAL__N_110tensorLoadEPNS_11ThreadStateEPKciiNS_8LocationE.exit, label %for.body.i, !llvm.loop !36

_ZN4gsan12_GLOBAL__N_110tensorLoadEPNS_11ThreadStateEPKciiNS_8LocationE.exit: ; preds = %if.end.i, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_init(ptr noundef %globalState, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i.i.i = add i64 %1, 39
  %cond.i.i.i.i = and i64 %ptr.biased.i.i.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val22.i.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i.i = zext i16 %globals.val22.i.i to i64
  %add.i.i.i = add nuw nsw i64 %conv.i.i.i, 1
  %conv1.i.i.i = zext i16 %globals.val.i.i to i64
  %mul.i.i.i = shl nuw nsw i64 %conv1.i.i.i, 1
  %mul3.i.i.i = mul nuw nsw i64 %mul.i.i.i, %add.i.i.i
  %add4.i.i.i = add nuw nsw i64 %mul3.i.i.i, 32
  %conv.i.i = zext i32 %0 to i64
  %mul.i.i = mul i64 %add4.i.i.i, %conv.i.i
  %add.i.i = add i64 %mul.i.i, %cond.i.i.i.i
  %4 = inttoptr i64 %add.i.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i.i = icmp eq ptr %5, null
  br i1 %cmp.i.i, label %if.then.i.i, label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

if.then.i.i:                                      ; preds = %entry
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase4.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase4.i.i, align 8, !tbaa !19
  %numReads.i.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i.i, align 8, !tbaa !3
  %clockBufferDirty.i.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i.i, align 4
  %globalsBase1.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i.i, align 8, !tbaa !20
  %sub.i.i.i = sub i64 %1, %7
  %div6.i.i.i = lshr i64 %sub.i.i.i, 30
  %numSms.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i.i, align 4, !tbaa !21
  %conv.i23.i.i = zext i16 %8 to i64
  %mul.i24.i.i = mul nuw nsw i64 %div6.i.i.i, %conv.i23.i.i
  %add.i25.i.i = add nuw nsw i64 %mul.i24.i.i, %conv.i.i
  %conv3.i.i.i = trunc i64 %add.i25.i.i to i16
  %threadId.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i.i, ptr %threadId.i.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i: ; preds = %if.then.i.i, %entry
  %9 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp.i = icmp eq i32 %9, 0
  br i1 %cmp.i, label %if.then.i, label %_ZN4gsan12_GLOBAL__N_110initThreadEPNS_11GlobalStateENS_8LocationE.exit

if.then.i:                                        ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i
  %globalsBase1.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %10 = load i64, ptr %globalsBase1.i.i, align 8, !tbaa !20
  %sub.i.i = sub i64 %1, %10
  %div6.i.i = lshr i64 %sub.i.i, 30
  %numSms.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %11 = load i16, ptr %numSms.i.i, align 4, !tbaa !21
  %conv.i16.i = zext i16 %11 to i64
  %mul.i17.i = mul nuw nsw i64 %div6.i.i, %conv.i16.i
  %add.i18.i = add nuw nsw i64 %mul.i17.i, %conv.i.i
  %vectorClock.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %idxprom.i = and i64 %add.i18.i, 65535
  %arrayidx.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i, i64 %idxprom.i
  %12 = load i16, ptr %arrayidx.i, align 2, !tbaa !22
  %cmp4.not.i = icmp eq i16 %12, -1
  br i1 %cmp4.not.i, label %if.then5.i, label %do.end.i

if.then5.i:                                       ; preds = %if.then.i
  %cmp.i19.i = icmp eq ptr %file, null
  %cond.i.i = select i1 %cmp.i19.i, ptr @.str, ptr %file
  tail call void @__assertfail(ptr noundef nonnull @.str6, ptr noundef nonnull %cond.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i = load i16, ptr %arrayidx.i, align 2, !tbaa !22
  br label %do.end.i

do.end.i:                                         ; preds = %if.then5.i, %if.then.i
  %13 = phi i16 [ %.pre.i, %if.then5.i ], [ %12, %if.then.i ]
  %add.i = add i16 %13, 1
  store i16 %add.i, ptr %arrayidx.i, align 2, !tbaa !22
  %clockBufferDirty.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  %bf.load.i = load i32, ptr %clockBufferDirty.i, align 4
  %bf.set.i = or i32 %bf.load.i, 1
  store i32 %bf.set.i, ptr %clockBufferDirty.i, align 4
  br label %_ZN4gsan12_GLOBAL__N_110initThreadEPNS_11GlobalStateENS_8LocationE.exit

_ZN4gsan12_GLOBAL__N_110initThreadEPNS_11GlobalStateENS_8LocationE.exit: ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i, %do.end.i
  ret void
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_store_tensor(ptr noundef %globalState, ptr noundef readonly captures(none) %stackPtr, i32 noundef %numElems, i32 noundef %bytesPerElem, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i.i = add i64 %1, 39
  %cond.i.i.i = and i64 %ptr.biased.i.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val22.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i = zext i16 %globals.val22.i to i64
  %add.i.i = add nuw nsw i64 %conv.i.i, 1
  %conv1.i.i = zext i16 %globals.val.i to i64
  %mul.i.i = shl nuw nsw i64 %conv1.i.i, 1
  %mul3.i.i = mul nuw nsw i64 %mul.i.i, %add.i.i
  %add4.i.i = add nuw nsw i64 %mul3.i.i, 32
  %conv.i = zext i32 %0 to i64
  %mul.i = mul i64 %add4.i.i, %conv.i
  %add.i = add i64 %mul.i, %cond.i.i.i
  %4 = inttoptr i64 %add.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i = icmp eq ptr %5, null
  br i1 %cmp.i, label %if.then.i, label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

if.then.i:                                        ; preds = %entry
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase4.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase4.i, align 8, !tbaa !19
  %numReads.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i, align 8, !tbaa !3
  %clockBufferDirty.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i, align 4
  %globalsBase1.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i, align 8, !tbaa !20
  %sub.i.i = sub i64 %1, %7
  %div6.i.i = lshr i64 %sub.i.i, 30
  %numSms.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i, align 4, !tbaa !21
  %conv.i23.i = zext i16 %8 to i64
  %mul.i24.i = mul nuw nsw i64 %div6.i.i, %conv.i23.i
  %add.i25.i = add nuw nsw i64 %mul.i24.i, %conv.i
  %conv3.i.i = trunc i64 %add.i25.i to i16
  %threadId.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i, ptr %threadId.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit: ; preds = %entry, %if.then.i
  %conv.i4 = sext i32 %numElems to i64
  %mul.i5 = shl nsw i64 %conv.i4, 3
  %add.ptr.i = getelementptr inbounds nuw i8, ptr %stackPtr, i64 %mul.i5
  %cmp9.i = icmp sgt i32 %numElems, 0
  br i1 %cmp9.i, label %for.body.lr.ph.i, label %_ZN4gsan12_GLOBAL__N_111tensorStoreEPNS_11ThreadStateEPKciiNS_8LocationE.exit

for.body.lr.ph.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  %conv.i.i6 = sext i32 %bytesPerElem to i64
  %reserveBase1.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  %lock.i.i = getelementptr inbounds nuw i8, ptr %4, i64 24
  %and.i.i.i.i20.i.i.i = and i64 %add.i, -1073741824
  %9 = inttoptr i64 %and.i.i.i.i20.i.i.i to ptr
  %numSms.i.i.i.i22.i.i.i = getelementptr inbounds nuw i8, ptr %9, i64 20
  %globalsBase.i.i.i.i25.i.i.i = getelementptr inbounds nuw i8, ptr %9, i64 8
  %10 = getelementptr i8, ptr %9, i64 24
  %11 = getelementptr i8, ptr %9, i64 26
  %vectorClock.i.i.i71.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %threadId.i.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  %cmp.i37.i.i.i.i99.i.i.i = icmp eq ptr %file, null
  %cond.i38.i.i.i.i100.i.i.i = select i1 %cmp.i37.i.i.i.i99.i.i.i, ptr @.str, ptr %file
  br label %for.body.i

for.body.i:                                       ; preds = %if.end.i, %for.body.lr.ph.i
  %i.010.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %if.end.i ]
  %idxprom.i = zext nneg i32 %i.010.i to i64
  %arrayidx2.i = getelementptr inbounds nuw i8, ptr %add.ptr.i, i64 %idxprom.i
  %12 = load i8, ptr %arrayidx2.i, align 1, !tbaa !23
  %tobool.not.i = icmp eq i8 %12, 0
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i7

if.then.i7:                                       ; preds = %for.body.i
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %stackPtr, i64 %idxprom.i
  %13 = load i64, ptr %arrayidx.i, align 8, !tbaa !19
  %add.i.i8 = add i64 %13, %conv.i.i6
  %sub.i.i.i = and i64 %13, -4
  %rem3.i.i.i = and i64 %add.i.i8, 3
  %cmp.i.i.i = icmp eq i64 %rem3.i.i.i, 0
  %sub5.i.i.i = sub nuw nsw i64 4, %rem3.i.i.i
  %cond.i.i.i9 = select i1 %cmp.i.i.i, i64 0, i64 %sub5.i.i.i
  %add.i.i.i = add i64 %cond.i.i.i9, %add.i.i8
  %14 = load i64, ptr %reserveBase1.i.i, align 8, !tbaa !19
  %15 = atomicrmw add ptr %lock.i.i, i32 1 syncscope("block") acquire, align 4
  %cmp.i19.i.i = icmp sgt i32 %15, -1
  br i1 %cmp.i19.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i

do.body.i.i.i:                                    ; preds = %if.then.i7, %do.body.i.i.i
  %16 = load atomic i32, ptr %lock.i.i syncscope("block") acquire, align 4
  %cmp3.not.i.i.i = icmp sgt i32 %16, -1
  br i1 %cmp3.not.i.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i, !llvm.loop !24

_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i: ; preds = %do.body.i.i.i, %if.then.i7
  %cmp26.i.i = icmp ult i64 %sub.i.i.i, %add.i.i.i
  br i1 %cmp26.i.i, label %for.body.i.i.preheader, label %_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i

for.body.i.i.preheader:                           ; preds = %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %invariant.op = sub i64 -549755813888, %14
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.preheader, %for.inc.i.i
  %addr.027.i.i = phi i64 [ %add8.i.i, %for.inc.i.i ], [ %sub.i.i.i, %for.body.i.i.preheader ]
  %and.i.i.i.i = and i64 %addr.027.i.i, -1099511627776
  %cmp.i20.i.i = icmp eq i64 %and.i.i.i.i, %14
  br i1 %cmp.i20.i.i, label %if.end.i.i, label %for.inc.i.i

if.end.i.i:                                       ; preds = %for.body.i.i
  %sub.i22.reass.i.reass.i.reass.reass = add i64 %addr.027.i.i, %invariant.op
  %div4.i.i.i = lshr exact i64 %sub.i22.reass.i.reass.i.reass.reass, 2
  %mul.i.i.i = mul i64 %div4.i.i.i, 24
  %add.i23.i.i = add i64 %mul.i.i.i, %14
  %17 = inttoptr i64 %add.i23.i.i to ptr
  %lock.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 22
  br label %while.cond.i.i.i

while.cond.i.i.i:                                 ; preds = %while.cond.i.i.i, %if.end.i.i
  %18 = cmpxchg weak ptr %lock.i.i.i, i16 0, i16 1 acquire monotonic, align 2
  %19 = extractvalue { i16, i1 } %18, 1
  br i1 %19, label %for.body.i.i.i, label %while.cond.i.i.i, !llvm.loop !26

for.cond.cleanup.i.i.i:                           ; preds = %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i
  %writeClock.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 16
  %20 = load i16, ptr %writeClock.i.i.i, align 4, !tbaa !27
  %cmp.i.i.i.i = icmp eq i16 %20, 0
  br i1 %cmp.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i, label %if.end.i.i.i.i

if.end.i.i.i.i:                                   ; preds = %for.cond.cleanup.i.i.i
  %.phi.trans.insert.i.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 18
  %clock.val14.i.pre.i.i.i.i = load i16, ptr %.phi.trans.insert.i.i.i.i, align 2
  %21 = and i16 %clock.val14.i.pre.i.i.i.i, 16384
  %bf.cast.not.i.i.i.i.i.i = icmp eq i16 %21, 0
  br i1 %bf.cast.not.i.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i, label %if.end.i.i.i.i.i.i

if.end.i.i.i.i.i.i:                               ; preds = %if.end.i.i.i.i
  %bf.clear2.i.i.i.i.i.i = and i16 %clock.val14.i.pre.i.i.i.i, 4095
  %22 = load i16, ptr %numSms.i.i.i.i22.i.i.i, align 4, !tbaa !21
  %bf.clear2.i.i.i.i.i.i.frozen = freeze i16 %bf.clear2.i.i.i.i.i.i
  %.frozen = freeze i16 %22
  %div16.i.i.i.i.i.i.i = udiv i16 %bf.clear2.i.i.i.i.i.i.frozen, %.frozen
  %23 = mul i16 %div16.i.i.i.i.i.i.i, %.frozen
  %rem17.i.i.i.i.i.i.i.decomposed = sub i16 %bf.clear2.i.i.i.i.i.i.frozen, %23
  %24 = load i64, ptr %globalsBase.i.i.i.i25.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i.i.i.i = zext nneg i16 %div16.i.i.i.i.i.i.i to i64
  %mul.i.i.i.i.i.i.i = shl nuw nsw i64 %conv5.i.i.i.i.i.i.i, 30
  %add.i.i.i.i.i.i.i = or disjoint i64 %mul.i.i.i.i.i.i.i, 39
  %ptr.biased.i.i.i.i.i.i.i.i.i = add i64 %add.i.i.i.i.i.i.i, %24
  %cond.i.i.i.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i.i.i.i = load i16, ptr %10, align 8, !tbaa !11
  %globals.val15.i.i.i.i.i.i.i = load i16, ptr %11, align 2, !tbaa !15
  %conv.i.i.i.i.i.i.i.i = zext i16 %globals.val15.i.i.i.i.i.i.i to i64
  %add.i.i.i.i.i.i.i.i = add nuw nsw i64 %conv.i.i.i.i.i.i.i.i, 1
  %conv1.i.i.i.i.i.i.i.i = zext i16 %globals.val.i.i.i.i.i.i.i to i64
  %mul.i.i.i.i.i.i.i.i = shl nuw nsw i64 %conv1.i.i.i.i.i.i.i.i, 1
  %mul3.i.i.i.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i.i.i.i.i, %add.i.i.i.i.i.i.i.i
  %add4.i.i.i.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i.i.i.i, 32
  %conv7.i.i.i.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i.i, %conv7.i.i.i.i.i.i.i
  %add9.i.i.i.i.i.i.i = add i64 %mul8.i.i.i.i.i.i.i, %cond.i.i.i.i.i.i.i.i.i
  %25 = inttoptr i64 %add9.i.i.i.i.i.i.i to ptr
  %conv.i.i.i.i.i.i.i = zext i16 %20 to i32
  %clockBufferHead.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %25, i64 20
  %bf.load.i.i.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i.i.i, align 4
  %bf.lshr.i.i.i.i.i.i.i = lshr i32 %bf.load.i.i.i.i.i.i.i, 1
  %cmp3.not.i.i.i.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i
  br i1 %cmp3.not.i.i.i.i.i.i.i, label %if.then4.i.i.i.i.i.i.i, label %do.end9.i.i.i.i.i.i.i

if.then4.i.i.i.i.i.i.i:                           ; preds = %if.end.i.i.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i.i100.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i.i.i, align 4
  %.pre42.i.i.i.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i.i.i.i, 1
  br label %do.end9.i.i.i.i.i.i.i

do.end9.i.i.i.i.i.i.i:                            ; preds = %if.then4.i.i.i.i.i.i.i, %if.end.i.i.i.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i.i.i.i, %if.end.i.i.i.i.i.i ], [ %.pre42.i.i.i.i.i.i.i, %if.then4.i.i.i.i.i.i.i ]
  %and.i.i.i.i.i.i.i.i = and i64 %add9.i.i.i.i.i.i.i, -1073741824
  %26 = inttoptr i64 %and.i.i.i.i.i.i.i.i to ptr
  %sub.i.i.i.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i
  %clockBufferSize.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %26, i64 26
  %27 = load i16, ptr %clockBufferSize.i.i.i.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i.i.i.i = zext i16 %27 to i32
  %cmp17.i.i.i.i.i.i.i = icmp slt i32 %sub.i.i.i.i.i.i.i, %conv16.i.i.i.i.i.i.i
  br i1 %cmp17.i.i.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i, label %if.then18.i.i.i.i.i.i.i

if.then18.i.i.i.i.i.i.i:                          ; preds = %do.end9.i.i.i.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i38.i.i.i.i100.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i: ; preds = %if.then18.i.i.i.i.i.i.i, %do.end9.i.i.i.i.i.i.i
  %28 = phi i16 [ %.pre.i.i.i.i.i.i.i, %if.then18.i.i.i.i.i.i.i ], [ %27, %do.end9.i.i.i.i.i.i.i ]
  %29 = urem i16 %20, %28
  %rem.i.i.i.i.i.i.i = zext i16 %29 to i64
  %vectorClock.i.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %25, i64 30
  %numThreads.i.i.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %26, i64 24
  %30 = load i16, ptr %numThreads.i.i.i.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i.i.i.i = zext i16 %30 to i64
  %add.ptr.i.i.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i.i.i.i
  %mul.i8.i.i.i.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i.i.i.i, %rem.i.i.i.i.i.i.i
  %add.ptr.i.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i.i.i.i, i64 %mul.i8.i.i.i.i.i.i
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i, %if.end.i.i.i.i
  %retval.0.i.i22.i.i.i.i = phi ptr [ %add.ptr.i.i.i.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i.i.i ], [ null, %if.end.i.i.i.i ]
  %tobool.not.not.i.i.i.i.i = icmp eq ptr %retval.0.i.i22.i.i.i.i, null
  br i1 %tobool.not.not.i.i.i.i.i, label %cleanup.i.i.i.i.i, label %if.then1.i.i.i.i.i

if.then1.i.i.i.i.i:                               ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i
  %31 = load i16, ptr %10, align 8, !tbaa !11
  %conv.i.i.i.i.i.i = zext i16 %31 to i32
  %cmp.not11.i.i.i.i.i.i = icmp eq i16 %31, 0
  br i1 %cmp.not11.i.i.i.i.i.i, label %cleanup.i.i.i.i.i, label %for.body.i.i.i.i.i.i

for.body.i.i.i.i.i.i:                             ; preds = %if.then1.i.i.i.i.i, %for.body.i.i.i.i.i.i
  %i.012.i.i.i.i.i.i = phi i32 [ %inc.i.i.i.i.i.i, %for.body.i.i.i.i.i.i ], [ 0, %if.then1.i.i.i.i.i ]
  %idxprom.i.i.i.i.i.i = zext nneg i32 %i.012.i.i.i.i.i.i to i64
  %arrayidx.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i71.i.i.i, i64 %idxprom.i.i.i.i.i.i
  %32 = load i16, ptr %arrayidx.i.i.i.i.i.i, align 2, !tbaa !22
  %arrayidx3.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i.i.i.i, i64 %idxprom.i.i.i.i.i.i
  %33 = load i16, ptr %arrayidx3.i.i.i.i.i.i, align 2, !tbaa !22
  %cmp5.i.i.i.i.i.i = icmp uge i16 %32, %33
  %inc.i.i.i.i.i.i = add nuw nsw i32 %i.012.i.i.i.i.i.i, 1
  %exitcond.not.i.i.i.i.i.i = icmp ne i32 %inc.i.i.i.i.i.i, %conv.i.i.i.i.i.i
  %or.cond.not.i.i.i.i.i.i = select i1 %cmp5.i.i.i.i.i.i, i1 %exitcond.not.i.i.i.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i, label %cleanup.i.i.i.i.i, !llvm.loop !31

cleanup.i.i.i.i.i:                                ; preds = %for.body.i.i.i.i.i.i, %if.then1.i.i.i.i.i, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i
  %retval.0.i23.i.i.i.i = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i.i ], [ true, %if.then1.i.i.i.i.i ], [ %cmp5.i.i.i.i.i.i, %for.body.i.i.i.i.i.i ]
  br i1 %tobool.not.not.i.i.i.i.i, label %cleanup.cont.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i

cleanup.cont.i.i.i.i.i:                           ; preds = %cleanup.i.i.i.i.i
  %bf.load.i.i.i.i.i = load i16, ptr %.phi.trans.insert.i.i.i.i, align 2
  %bf.clear.i.i.i.i.i = and i16 %bf.load.i.i.i.i.i, 4095
  %idxprom.i.i.i.i.i = zext nneg i16 %bf.clear.i.i.i.i.i to i64
  %arrayidx.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i71.i.i.i, i64 %idxprom.i.i.i.i.i
  %34 = load i16, ptr %arrayidx.i.i.i.i.i, align 2, !tbaa !22
  %35 = load i16, ptr %writeClock.i.i.i, align 4, !tbaa !27
  %cmp7.i.i.i.i.i = icmp uge i16 %34, %35
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i: ; preds = %cleanup.cont.i.i.i.i.i, %cleanup.i.i.i.i.i
  %retval.1.i.i.i.i.i = phi i1 [ %retval.0.i23.i.i.i.i, %cleanup.i.i.i.i.i ], [ %cmp7.i.i.i.i.i, %cleanup.cont.i.i.i.i.i ]
  br i1 %retval.1.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i, label %if.then9.i.i.i.i

if.then9.i.i.i.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str8, ptr noundef nonnull %cond.i38.i.i.i.i100.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

for.body.i.i.i:                                   ; preds = %while.cond.i.i.i, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i
  %iRead.0104.i.i.i = phi i32 [ %inc.i.i.i, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i ], [ 0, %while.cond.i.i.i ]
  %idxprom.i.i.i = zext nneg i32 %iRead.0104.i.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds nuw [4 x i8], ptr %17, i64 %idxprom.i.i.i
  %36 = load i16, ptr %arrayidx.i.i.i, align 4, !tbaa !27
  %cmp.i11.i.i.i = icmp eq i16 %36, 0
  br i1 %cmp.i11.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i, label %if.end.i12.i.i.i

if.end.i12.i.i.i:                                 ; preds = %for.body.i.i.i
  %.phi.trans.insert.i13.i.i.i = getelementptr inbounds nuw i8, ptr %arrayidx.i.i.i, i64 2
  %clock.val14.i.pre.i14.i.i.i = load i16, ptr %.phi.trans.insert.i13.i.i.i, align 2
  %37 = and i16 %clock.val14.i.pre.i14.i.i.i, 16384
  %bf.cast.not.i.i.i18.i.i.i = icmp eq i16 %37, 0
  br i1 %bf.cast.not.i.i.i18.i.i.i, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i66.i.i.i, label %if.end.i.i.i19.i.i.i

if.end.i.i.i19.i.i.i:                             ; preds = %if.end.i12.i.i.i
  %bf.clear2.i.i.i21.i.i.i = and i16 %clock.val14.i.pre.i14.i.i.i, 4095
  %38 = load i16, ptr %numSms.i.i.i.i22.i.i.i, align 4, !tbaa !21
  %bf.clear2.i.i.i21.i.i.i.frozen = freeze i16 %bf.clear2.i.i.i21.i.i.i
  %.frozen12 = freeze i16 %38
  %div16.i.i.i.i23.i.i.i = udiv i16 %bf.clear2.i.i.i21.i.i.i.frozen, %.frozen12
  %39 = mul i16 %div16.i.i.i.i23.i.i.i, %.frozen12
  %rem17.i.i.i.i24.i.i.i.decomposed = sub i16 %bf.clear2.i.i.i21.i.i.i.frozen, %39
  %40 = load i64, ptr %globalsBase.i.i.i.i25.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i26.i.i.i = zext nneg i16 %div16.i.i.i.i23.i.i.i to i64
  %mul.i.i.i.i27.i.i.i = shl nuw nsw i64 %conv5.i.i.i.i26.i.i.i, 30
  %add.i.i.i.i28.i.i.i = or disjoint i64 %mul.i.i.i.i27.i.i.i, 39
  %ptr.biased.i.i.i.i.i.i29.i.i.i = add i64 %add.i.i.i.i28.i.i.i, %40
  %cond.i.i.i.i.i.i30.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i29.i.i.i, -8
  %globals.val.i.i.i.i31.i.i.i = load i16, ptr %10, align 8, !tbaa !11
  %globals.val15.i.i.i.i32.i.i.i = load i16, ptr %11, align 2, !tbaa !15
  %conv.i.i.i.i.i33.i.i.i = zext i16 %globals.val15.i.i.i.i32.i.i.i to i64
  %add.i.i.i.i.i34.i.i.i = add nuw nsw i64 %conv.i.i.i.i.i33.i.i.i, 1
  %conv1.i.i.i.i.i35.i.i.i = zext i16 %globals.val.i.i.i.i31.i.i.i to i64
  %mul.i.i.i.i.i36.i.i.i = shl nuw nsw i64 %conv1.i.i.i.i.i35.i.i.i, 1
  %mul3.i.i.i.i.i37.i.i.i = mul nuw nsw i64 %mul.i.i.i.i.i36.i.i.i, %add.i.i.i.i.i34.i.i.i
  %add4.i.i.i.i.i38.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i37.i.i.i, 32
  %conv7.i.i.i.i39.i.i.i = zext nneg i16 %rem17.i.i.i.i24.i.i.i.decomposed to i64
  %mul8.i.i.i.i40.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i38.i.i.i, %conv7.i.i.i.i39.i.i.i
  %add9.i.i.i.i41.i.i.i = add i64 %mul8.i.i.i.i40.i.i.i, %cond.i.i.i.i.i.i30.i.i.i
  %41 = inttoptr i64 %add9.i.i.i.i41.i.i.i to ptr
  %conv.i.i.i.i42.i.i.i = zext i16 %36 to i32
  %clockBufferHead.i.i.i.i43.i.i.i = getelementptr inbounds nuw i8, ptr %41, i64 20
  %bf.load.i.i.i.i44.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i43.i.i.i, align 4
  %bf.lshr.i.i.i.i45.i.i.i = lshr i32 %bf.load.i.i.i.i44.i.i.i, 1
  %cmp3.not.i.i.i.i46.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i45.i.i.i, %conv.i.i.i.i42.i.i.i
  br i1 %cmp3.not.i.i.i.i46.i.i.i, label %if.then4.i.i.i.i98.i.i.i, label %do.end9.i.i.i.i47.i.i.i

if.then4.i.i.i.i98.i.i.i:                         ; preds = %if.end.i.i.i19.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i.i100.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i101.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i43.i.i.i, align 4
  %.pre42.i.i.i.i102.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i101.i.i.i, 1
  br label %do.end9.i.i.i.i47.i.i.i

do.end9.i.i.i.i47.i.i.i:                          ; preds = %if.then4.i.i.i.i98.i.i.i, %if.end.i.i.i19.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i48.i.i.i = phi i32 [ %bf.lshr.i.i.i.i45.i.i.i, %if.end.i.i.i19.i.i.i ], [ %.pre42.i.i.i.i102.i.i.i, %if.then4.i.i.i.i98.i.i.i ]
  %and.i.i.i.i.i49.i.i.i = and i64 %add9.i.i.i.i41.i.i.i, -1073741824
  %42 = inttoptr i64 %and.i.i.i.i.i49.i.i.i to ptr
  %sub.i.i.i.i50.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i48.i.i.i, %conv.i.i.i.i42.i.i.i
  %clockBufferSize.i.i.i.i51.i.i.i = getelementptr inbounds nuw i8, ptr %42, i64 26
  %43 = load i16, ptr %clockBufferSize.i.i.i.i51.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i52.i.i.i = zext i16 %43 to i32
  %cmp17.i.i.i.i53.i.i.i = icmp slt i32 %sub.i.i.i.i50.i.i.i, %conv16.i.i.i.i52.i.i.i
  br i1 %cmp17.i.i.i.i53.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i58.i.i.i, label %if.then18.i.i.i.i54.i.i.i

if.then18.i.i.i.i54.i.i.i:                        ; preds = %do.end9.i.i.i.i47.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i38.i.i.i.i100.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i57.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i51.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i58.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i58.i.i.i: ; preds = %if.then18.i.i.i.i54.i.i.i, %do.end9.i.i.i.i47.i.i.i
  %44 = phi i16 [ %.pre.i.i.i.i57.i.i.i, %if.then18.i.i.i.i54.i.i.i ], [ %43, %do.end9.i.i.i.i47.i.i.i ]
  %45 = urem i16 %36, %44
  %rem.i.i.i.i59.i.i.i = zext i16 %45 to i64
  %vectorClock.i.i.i.i.i60.i.i.i = getelementptr inbounds nuw i8, ptr %41, i64 30
  %numThreads.i.i.i.i.i61.i.i.i = getelementptr inbounds nuw i8, ptr %42, i64 24
  %46 = load i16, ptr %numThreads.i.i.i.i.i61.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i62.i.i.i = zext i16 %46 to i64
  %add.ptr.i.i.i.i.i63.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i60.i.i.i, i64 %idx.ext.i.i.i.i.i62.i.i.i
  %mul.i8.i.i.i64.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i62.i.i.i, %rem.i.i.i.i59.i.i.i
  %add.ptr.i.i.i.i65.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i63.i.i.i, i64 %mul.i8.i.i.i64.i.i.i
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i66.i.i.i

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i66.i.i.i: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i58.i.i.i, %if.end.i12.i.i.i
  %retval.0.i.i22.i67.i.i.i = phi ptr [ %add.ptr.i.i.i.i65.i.i.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i58.i.i.i ], [ null, %if.end.i12.i.i.i ]
  %tobool.not.not.i.i68.i.i.i = icmp eq ptr %retval.0.i.i22.i67.i.i.i, null
  br i1 %tobool.not.not.i.i68.i.i.i, label %cleanup.i.i84.i.i.i, label %if.then1.i.i69.i.i.i

if.then1.i.i69.i.i.i:                             ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i66.i.i.i
  %47 = load i16, ptr %10, align 8, !tbaa !11
  %conv.i.i.i73.i.i.i = zext i16 %47 to i32
  %cmp.not11.i.i.i74.i.i.i = icmp eq i16 %47, 0
  br i1 %cmp.not11.i.i.i74.i.i.i, label %cleanup.i.i84.i.i.i, label %for.body.i.i.i75.i.i.i

for.body.i.i.i75.i.i.i:                           ; preds = %if.then1.i.i69.i.i.i, %for.body.i.i.i75.i.i.i
  %i.012.i.i.i76.i.i.i = phi i32 [ %inc.i.i.i81.i.i.i, %for.body.i.i.i75.i.i.i ], [ 0, %if.then1.i.i69.i.i.i ]
  %idxprom.i.i.i77.i.i.i = zext nneg i32 %i.012.i.i.i76.i.i.i to i64
  %arrayidx.i.i.i78.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i71.i.i.i, i64 %idxprom.i.i.i77.i.i.i
  %48 = load i16, ptr %arrayidx.i.i.i78.i.i.i, align 2, !tbaa !22
  %arrayidx3.i.i.i79.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i67.i.i.i, i64 %idxprom.i.i.i77.i.i.i
  %49 = load i16, ptr %arrayidx3.i.i.i79.i.i.i, align 2, !tbaa !22
  %cmp5.i.i.i80.i.i.i = icmp uge i16 %48, %49
  %inc.i.i.i81.i.i.i = add nuw nsw i32 %i.012.i.i.i76.i.i.i, 1
  %exitcond.not.i.i.i82.i.i.i = icmp ne i32 %inc.i.i.i81.i.i.i, %conv.i.i.i73.i.i.i
  %or.cond.not.i.i.i83.i.i.i = select i1 %cmp5.i.i.i80.i.i.i, i1 %exitcond.not.i.i.i82.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i83.i.i.i, label %for.body.i.i.i75.i.i.i, label %cleanup.i.i84.i.i.i, !llvm.loop !31

cleanup.i.i84.i.i.i:                              ; preds = %for.body.i.i.i75.i.i.i, %if.then1.i.i69.i.i.i, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i66.i.i.i
  %retval.0.i23.i85.i.i.i = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i66.i.i.i ], [ true, %if.then1.i.i69.i.i.i ], [ %cmp5.i.i.i80.i.i.i, %for.body.i.i.i75.i.i.i ]
  br i1 %tobool.not.not.i.i68.i.i.i, label %cleanup.cont.i.i91.i.i.i, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i86.i.i.i

cleanup.cont.i.i91.i.i.i:                         ; preds = %cleanup.i.i84.i.i.i
  %bf.load.i.i93.i.i.i = load i16, ptr %.phi.trans.insert.i13.i.i.i, align 2
  %bf.clear.i.i94.i.i.i = and i16 %bf.load.i.i93.i.i.i, 4095
  %idxprom.i.i95.i.i.i = zext nneg i16 %bf.clear.i.i94.i.i.i to i64
  %arrayidx.i.i96.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i71.i.i.i, i64 %idxprom.i.i95.i.i.i
  %50 = load i16, ptr %arrayidx.i.i96.i.i.i, align 2, !tbaa !22
  %51 = load i16, ptr %arrayidx.i.i.i, align 4, !tbaa !27
  %cmp7.i.i97.i.i.i = icmp uge i16 %50, %51
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i86.i.i.i

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i86.i.i.i: ; preds = %cleanup.cont.i.i91.i.i.i, %cleanup.i.i84.i.i.i
  %retval.1.i.i87.i.i.i = phi i1 [ %retval.0.i23.i85.i.i.i, %cleanup.i.i84.i.i.i ], [ %cmp7.i.i97.i.i.i, %cleanup.cont.i.i91.i.i.i ]
  br i1 %retval.1.i.i87.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i, label %if.then9.i88.i.i.i

if.then9.i88.i.i.i:                               ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i86.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str7, ptr noundef nonnull %cond.i38.i.i.i.i100.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i

_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit103.i.i.i: ; preds = %if.then9.i88.i.i.i, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i86.i.i.i, %for.body.i.i.i
  %inc.i.i.i = add nuw nsw i32 %iRead.0104.i.i.i, 1
  %exitcond.not.i.i.i = icmp eq i32 %inc.i.i.i, 4
  br i1 %exitcond.not.i.i.i, label %for.cond.cleanup.i.i.i, label %for.body.i.i.i, !llvm.loop !37

_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i: ; preds = %if.then9.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i.i, %for.cond.cleanup.i.i.i
  %52 = load i16, ptr %threadId.i.i.i.i, align 4, !tbaa !22
  %idxprom.i.i.i.i = zext i16 %52 to i64
  %arrayidx.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i71.i.i.i, i64 %idxprom.i.i.i.i
  %53 = load i16, ptr %arrayidx.i.i.i.i, align 2, !tbaa !22
  %bf.value.i.i.i.i = and i16 %52, 4095
  store i16 %53, ptr %writeClock.i.i.i, align 4, !tbaa !22
  %ref.tmp.sroa.4.0.writeClock2.sroa_idx.i.i.i = getelementptr inbounds nuw i8, ptr %17, i64 18
  store i16 %bf.value.i.i.i.i, ptr %ref.tmp.sroa.4.0.writeClock2.sroa_idx.i.i.i, align 2, !tbaa !23
  store atomic i16 0, ptr %lock.i.i.i release, align 2
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i, %for.body.i.i
  %add8.i.i = add i64 %addr.027.i.i, 4
  %cmp.i.i = icmp ult i64 %add8.i.i, %add.i.i.i
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, !llvm.loop !38

_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i: ; preds = %for.inc.i.i, %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %54 = atomicrmw sub ptr %lock.i.i, i32 1 syncscope("block") monotonic, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, %for.body.i
  %inc.i = add nuw nsw i32 %i.010.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, %numElems
  br i1 %exitcond.not.i, label %_ZN4gsan12_GLOBAL__N_111tensorStoreEPNS_11ThreadStateEPKciiNS_8LocationE.exit, label %for.body.i, !llvm.loop !39

_ZN4gsan12_GLOBAL__N_111tensorStoreEPNS_11ThreadStateEPKciiNS_8LocationE.exit: ; preds = %if.end.i, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  ret void
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_atomic_tensor(ptr noundef %globalState, ptr noundef readonly captures(none) %stackPtr, i32 noundef %numElems, i32 noundef %bytesPerElem, i32 noundef %sem, i32 noundef %scope, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %event = alloca %"struct.gsan::AtomicEventState", align 8
  %agg.tmp5 = alloca %"struct.gsan::Location", align 8
  %conv = sext i32 %numElems to i64
  %mul = shl nsw i64 %conv, 3
  %add.ptr = getelementptr inbounds nuw i8, ptr %stackPtr, i64 %mul
  %cmp20 = icmp sgt i32 %numElems, 0
  br i1 %cmp20, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i.i.i = add i64 %1, 39
  %cond.i.i.i.i = and i64 %ptr.biased.i.i.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %3 = getelementptr i8, ptr %globalState, i64 26
  %conv.i.i = zext i32 %0 to i64
  %globalsBase1.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %numSms.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %conv.i46.i = sext i32 %bytesPerElem to i64
  %cmp.i54.i.i = icmp eq ptr %file, null
  %cond.i55.i.i = select i1 %cmp.i54.i.i, ptr @.str, ptr %file
  %numCells16.i.i = getelementptr inbounds nuw i8, ptr %event, i64 32
  %cells.i.i = getelementptr inbounds nuw i8, ptr %event, i64 8
  %switch.tableidx.i.i = add i32 %sem, -1
  %4 = icmp ult i32 %switch.tableidx.i.i, 4
  %switch.idx.cast.i.i = trunc nuw nsw i32 %switch.tableidx.i.i to i8
  %loc.sroa.5.0.agg.tmp5.sroa_idx = getelementptr inbounds nuw i8, ptr %agg.tmp5, i64 8
  %switch.tableidx = add i32 %scope, -1
  %5 = icmp ult i32 %switch.tableidx, 3
  %switch.cast = trunc nuw i32 %switch.tableidx to i24
  %switch.shiftamt = shl nuw nsw i24 %switch.cast, 3
  %switch.downshift = lshr i24 196866, %switch.shiftamt
  %switch.masked = trunc i24 %switch.downshift to i8
  %6 = trunc i24 %switch.downshift to i16
  %7 = shl i16 %6, 12
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i.021 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %idxprom = zext nneg i32 %i.021 to i64
  %arrayidx = getelementptr inbounds nuw i8, ptr %add.ptr, i64 %idxprom
  %8 = load i8, ptr %arrayidx, align 1, !tbaa !23
  %tobool.not = icmp eq i8 %8, 0
  br i1 %tobool.not, label %for.inc, label %if.end

if.end:                                           ; preds = %for.body
  call void @llvm.lifetime.start.p0(ptr nonnull %event) #9
  %arrayidx4 = getelementptr inbounds nuw [8 x i8], ptr %stackPtr, i64 %idxprom
  %9 = load i64, ptr %arrayidx4, align 8, !tbaa !19
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(33) %event, i8 0, i64 33, i1 false)
  %globals.val.i.i = load i16, ptr %2, align 8, !tbaa !11
  %globals.val22.i.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i.i = zext i16 %globals.val22.i.i to i64
  %add.i.i.i = add nuw nsw i64 %conv.i.i.i, 1
  %conv1.i.i.i = zext i16 %globals.val.i.i to i64
  %mul.i.i.i = shl nuw nsw i64 %conv1.i.i.i, 1
  %mul3.i.i.i = mul nuw nsw i64 %mul.i.i.i, %add.i.i.i
  %add4.i.i.i = add nuw nsw i64 %mul3.i.i.i, 32
  %mul.i.i = mul i64 %add4.i.i.i, %conv.i.i
  %add.i.i = add i64 %mul.i.i, %cond.i.i.i.i
  %10 = inttoptr i64 %add.i.i to ptr
  %11 = load ptr, ptr %10, align 8, !tbaa !16
  %cmp.i.i = icmp eq ptr %11, null
  br i1 %cmp.i.i, label %if.then.i.i, label %if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i

if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i: ; preds = %if.end
  %reserveBase1.i.phi.trans.insert.i = getelementptr inbounds nuw i8, ptr %10, i64 8
  %.pre.i = load i64, ptr %reserveBase1.i.phi.trans.insert.i, align 8, !tbaa !19
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

if.then.i.i:                                      ; preds = %if.end
  %12 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase4.i.i = getelementptr inbounds nuw i8, ptr %10, i64 8
  store i64 %12, ptr %reserveBase4.i.i, align 8, !tbaa !19
  %numReads.i.i = getelementptr inbounds nuw i8, ptr %10, i64 16
  store i32 0, ptr %numReads.i.i, align 8, !tbaa !3
  %clockBufferDirty.i.i = getelementptr inbounds nuw i8, ptr %10, i64 20
  store i32 0, ptr %clockBufferDirty.i.i, align 4
  %13 = load i64, ptr %globalsBase1.i.i.i, align 8, !tbaa !20
  %sub.i.i.i = sub i64 %1, %13
  %div6.i.i.i = lshr i64 %sub.i.i.i, 30
  %14 = load i16, ptr %numSms.i.i.i, align 4, !tbaa !21
  %conv.i23.i.i = zext i16 %14 to i64
  %mul.i24.i.i = mul nuw nsw i64 %div6.i.i.i, %conv.i23.i.i
  %add.i25.i.i = add nuw nsw i64 %mul.i24.i.i, %conv.i.i
  %conv3.i.i.i = trunc i64 %add.i25.i.i to i16
  %threadId.i.i = getelementptr inbounds nuw i8, ptr %10, i64 28
  store i16 %conv3.i.i.i, ptr %threadId.i.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %10, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i: ; preds = %if.then.i.i, %if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i
  %15 = phi i64 [ %.pre.i, %if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i ], [ %12, %if.then.i.i ]
  %add.i47.i = add i64 %9, %conv.i46.i
  %sub.i.i48.i = and i64 %9, -4
  %rem3.i.i.i = and i64 %add.i47.i, 3
  %cmp.i.i.i = icmp eq i64 %rem3.i.i.i, 0
  %sub5.i.i.i = sub nuw nsw i64 4, %rem3.i.i.i
  %cond.i.i.i = select i1 %cmp.i.i.i, i64 0, i64 %sub5.i.i.i
  %add.i.i49.i = add i64 %cond.i.i.i, %add.i47.i
  %cmp62.i.i = icmp ult i64 %sub.i.i48.i, %add.i.i49.i
  br i1 %cmp62.i.i, label %for.body.i.i.preheader, label %for.cond.cleanup.i.i

for.body.i.i.preheader:                           ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i
  %16 = add i64 %sub.i.i48.i, 4
  %umax = tail call i64 @llvm.umax.i64(i64 %add.i.i49.i, i64 %16)
  %17 = xor i64 %sub.i.i48.i, -1
  %18 = add i64 %umax, %17
  %19 = lshr i64 %18, 2
  %20 = add nuw nsw i64 %19, 1
  %min.iters.check = icmp eq i64 %19, 0
  br i1 %min.iters.check, label %for.body.i.i.preheader33, label %vector.ph

vector.ph:                                        ; preds = %for.body.i.i.preheader
  %n.vec = and i64 %20, 9223372036854775806
  %21 = shl i64 %n.vec, 2
  %22 = add i64 %sub.i.i48.i, %21
  %broadcast.splatinsert = insertelement <2 x i64> poison, i64 %15, i64 0
  %broadcast.splat = shufflevector <2 x i64> %broadcast.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert30 = insertelement <2 x i64> poison, i64 %sub.i.i48.i, i64 0
  %broadcast.splat31 = shufflevector <2 x i64> %broadcast.splatinsert30, <2 x i64> poison, <2 x i32> zeroinitializer
  %induction = add <2 x i64> %broadcast.splat31, <i64 0, i64 4>
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <2 x i64> [ %induction, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %vec.phi = phi <2 x i8> [ zeroinitializer, %vector.ph ], [ %26, %vector.body ]
  %23 = and <2 x i64> %vec.ind, splat (i64 -1099511627776)
  %24 = icmp eq <2 x i64> %23, %broadcast.splat
  %25 = zext <2 x i1> %24 to <2 x i8>
  %26 = add <2 x i8> %vec.phi, %25
  %index.next = add nuw i64 %index, 2
  %vec.ind.next = add <2 x i64> %vec.ind, splat (i64 8)
  %27 = icmp eq i64 %index.next, %n.vec
  br i1 %27, label %middle.block, label %vector.body, !llvm.loop !40

middle.block:                                     ; preds = %vector.body
  %28 = tail call i8 @llvm.vector.reduce.add.v2i8(<2 x i8> %26)
  %cmp.n = icmp eq i64 %20, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup.i.i, label %for.body.i.i.preheader33

for.body.i.i.preheader33:                         ; preds = %for.body.i.i.preheader, %middle.block
  %addr.064.i.i.ph = phi i64 [ %sub.i.i48.i, %for.body.i.i.preheader ], [ %22, %middle.block ]
  %numCells.063.i.i.ph = phi i8 [ 0, %for.body.i.i.preheader ], [ %28, %middle.block ]
  br label %for.body.i.i

for.cond.cleanup.i.i:                             ; preds = %for.body.i.i, %middle.block, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i
  %numCells.0.lcssa.i.i = phi i8 [ 0, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i ], [ %28, %middle.block ], [ %spec.select.i.i, %for.body.i.i ]
  %cmp7.i.i = icmp ult i8 %numCells.0.lcssa.i.i, 4
  br i1 %cmp7.i.i, label %do.end.i.i, label %if.then8.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.preheader33, %for.body.i.i
  %addr.064.i.i = phi i64 [ %add5.i.i, %for.body.i.i ], [ %addr.064.i.i.ph, %for.body.i.i.preheader33 ]
  %numCells.063.i.i = phi i8 [ %spec.select.i.i, %for.body.i.i ], [ %numCells.063.i.i.ph, %for.body.i.i.preheader33 ]
  %and.i.i.i.i = and i64 %addr.064.i.i, -1099511627776
  %cmp.i53.i.i = icmp eq i64 %and.i.i.i.i, %15
  %inc.i.i = zext i1 %cmp.i53.i.i to i8
  %spec.select.i.i = add i8 %numCells.063.i.i, %inc.i.i
  %add5.i.i = add i64 %addr.064.i.i, 4
  %cmp.i51.i = icmp ult i64 %add5.i.i, %add.i.i49.i
  br i1 %cmp.i51.i, label %for.body.i.i, label %for.cond.cleanup.i.i, !llvm.loop !43

if.then8.i.i:                                     ; preds = %for.cond.cleanup.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str9, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.end.i.i

do.end.i.i:                                       ; preds = %if.then8.i.i, %for.cond.cleanup.i.i
  %cmp13.i.i = icmp eq i8 %numCells.0.lcssa.i.i, 0
  br i1 %cmp13.i.i, label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i, label %if.end15.i.i

if.end15.i.i:                                     ; preds = %do.end.i.i
  %lock.i.i = getelementptr inbounds nuw i8, ptr %10, i64 24
  br label %while.cond.i.i.i

while.cond.i.i.i:                                 ; preds = %while.cond.i.i.i, %if.end15.i.i
  %29 = cmpxchg weak ptr %lock.i.i, i32 0, i32 -2147483648 syncscope("block") acquire monotonic, align 4
  %30 = extractvalue { i32, i1 } %29, 1
  br i1 %30, label %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i, label %while.cond.i.i.i, !llvm.loop !44

_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i: ; preds = %while.cond.i.i.i
  store ptr %10, ptr %event, align 8, !tbaa !45
  store i8 0, ptr %numCells16.i.i, align 8, !tbaa !48
  br i1 %cmp62.i.i, label %for.body23.i.i.preheader, label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i

for.body23.i.i.preheader:                         ; preds = %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i
  %invariant.op = sub i64 -549755813888, %15
  br label %for.body23.i.i

for.body23.i.i:                                   ; preds = %for.body23.i.i.preheader, %for.inc31.i.i
  %31 = phi i8 [ %36, %for.inc31.i.i ], [ 0, %for.body23.i.i.preheader ]
  %addr17.066.i.i = phi i64 [ %add32.i.i, %for.inc31.i.i ], [ %sub.i.i48.i, %for.body23.i.i.preheader ]
  %and.i.i56.i.i = and i64 %addr17.066.i.i, -1099511627776
  %cmp.i57.i.i = icmp eq i64 %and.i.i56.i.i, %15
  br i1 %cmp.i57.i.i, label %if.end26.i.i, label %for.inc31.i.i

if.end26.i.i:                                     ; preds = %for.body23.i.i
  %sub.i59.reass.i.reass.i.reass.reass = add i64 %addr17.066.i.i, %invariant.op
  %div4.i.i.i = lshr exact i64 %sub.i59.reass.i.reass.i.reass.reass, 2
  %mul.i.i50.i = mul i64 %div4.i.i.i, 24
  %add.i60.i.i = add i64 %mul.i.i50.i, %15
  %32 = inttoptr i64 %add.i60.i.i to ptr
  %lock.i.i.i = getelementptr inbounds nuw i8, ptr %32, i64 22
  br label %while.cond.i61.i.i

while.cond.i61.i.i:                               ; preds = %while.cond.i61.i.i, %if.end26.i.i
  %33 = cmpxchg weak ptr %lock.i.i.i, i16 0, i16 1 acquire monotonic, align 2
  %34 = extractvalue { i16, i1 } %33, 1
  br i1 %34, label %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, label %while.cond.i61.i.i, !llvm.loop !26

_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i: ; preds = %while.cond.i61.i.i
  %35 = load i8, ptr %numCells16.i.i, align 8, !tbaa !48
  %inc30.i.i = add i8 %35, 1
  store i8 %inc30.i.i, ptr %numCells16.i.i, align 8, !tbaa !48
  %idxprom.i.i = zext i8 %35 to i64
  %arrayidx.i.i = getelementptr inbounds nuw [8 x i8], ptr %cells.i.i, i64 %idxprom.i.i
  store ptr %32, ptr %arrayidx.i.i, align 8, !tbaa !49
  br label %for.inc31.i.i

for.inc31.i.i:                                    ; preds = %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, %for.body23.i.i
  %36 = phi i8 [ %inc30.i.i, %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i ], [ %31, %for.body23.i.i ]
  %add32.i.i = add i64 %addr17.066.i.i, 4
  %cmp21.i.i = icmp ult i64 %add32.i.i, %add.i.i49.i
  br i1 %cmp21.i.i, label %for.body23.i.i, label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i.loopexit, !llvm.loop !51

_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i.loopexit: ; preds = %for.inc31.i.i
  %.pre = load ptr, ptr %event, align 8, !tbaa !45
  %37 = icmp eq ptr %.pre, null
  br label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i

_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i: ; preds = %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i.loopexit, %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i, %do.end.i.i
  %38 = phi i8 [ %36, %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i.loopexit ], [ 0, %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i ], [ 0, %do.end.i.i ]
  %cmp.i = phi i1 [ %37, %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i.loopexit ], [ false, %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i ], [ true, %do.end.i.i ]
  br i1 %cmp.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, label %if.end2.i

if.end2.i:                                        ; preds = %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i
  br i1 %4, label %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i, label %sw.default.i.i

sw.default.i.i:                                   ; preds = %if.end2.i
  tail call void @llvm.trap()
  unreachable

_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i: ; preds = %if.end2.i
  br i1 %5, label %switch.lookup, label %sw.default.i52.i

sw.default.i52.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i
  tail call void @llvm.trap()
  unreachable

switch.lookup:                                    ; preds = %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i
  %cmp6113.not.i = icmp eq i8 %38, 0
  br i1 %cmp6113.not.i, label %for.cond.cleanup.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %switch.lookup
  %threadId.i56.i = getelementptr inbounds nuw i8, ptr %10, i64 28
  %and.i.i.i = and i64 %add.i.i, -1073741824
  %39 = inttoptr i64 %and.i.i.i to ptr
  %numSms.i.i.i.i = getelementptr inbounds nuw i8, ptr %39, i64 20
  %globalsBase.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %39, i64 8
  %40 = getelementptr i8, ptr %39, i64 24
  %41 = getelementptr i8, ptr %39, i64 26
  %vectorClock.i.i.i.i = getelementptr inbounds nuw i8, ptr %10, i64 30
  %numReads18.i.i = getelementptr inbounds nuw i8, ptr %10, i64 16
  %rngSeed.i.i = getelementptr inbounds nuw i8, ptr %39, i64 16
  br label %for.body.i

for.cond.cleanup.i:                               ; preds = %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i, %switch.lookup
  switch i8 %switch.idx.cast.i.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit [
    i8 3, label %for.cond11.preheader.i
    i8 1, label %for.cond11.preheader.i
  ]

for.cond11.preheader.i:                           ; preds = %for.cond.cleanup.i, %for.cond.cleanup.i
  br i1 %cmp6113.not.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, label %for.body17.lr.ph.i

for.body17.lr.ph.i:                               ; preds = %for.cond11.preheader.i
  %threadId.i71.i = getelementptr inbounds nuw i8, ptr %10, i64 28
  %and.i.i72.i = and i64 %add.i.i, -1073741824
  %42 = inttoptr i64 %and.i.i72.i to ptr
  %numSms.i.i.i105.i = getelementptr inbounds nuw i8, ptr %42, i64 20
  %globalsBase.i.i.i.i = getelementptr inbounds nuw i8, ptr %42, i64 8
  %43 = getelementptr i8, ptr %42, i64 24
  %44 = getelementptr i8, ptr %42, i64 26
  %vectorClock.i.i = getelementptr inbounds nuw i8, ptr %10, i64 30
  %clockBufferDirty.i95.i = getelementptr inbounds nuw i8, ptr %10, i64 20
  br label %for.body17.i

for.body.i:                                       ; preds = %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i, %for.body.lr.ph.i
  %i.0114.i = phi i8 [ 0, %for.body.lr.ph.i ], [ %inc.i, %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i ]
  %idxprom.i = zext i8 %i.0114.i to i64
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %cells.i.i, i64 %idxprom.i
  %45 = load ptr, ptr %arrayidx.i, align 8, !tbaa !49
  %writeClock.i = getelementptr inbounds nuw i8, ptr %45, i64 16
  %46 = load i32, ptr %writeClock.i, align 4
  %write.sroa.0.0.extract.trunc.i = trunc i32 %46 to i16
  %write.sroa.5.0.extract.shift.i = lshr i32 %46, 16
  %write.sroa.5.0.extract.trunc.i = trunc nuw i32 %write.sroa.5.0.extract.shift.i to i16
  %cmp.i55.i = icmp eq i16 %write.sroa.0.0.extract.trunc.i, 0
  br i1 %cmp.i55.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i, label %if.end.i.i

if.end.i.i:                                       ; preds = %for.body.i
  %47 = load i16, ptr %threadId.i56.i, align 4, !tbaa !22
  %bf.lshr.i.i = lshr i16 %write.sroa.5.0.extract.trunc.i, 12
  %48 = trunc nuw nsw i16 %bf.lshr.i.i to i8
  %bf.cast.i.i = and i8 %48, 3
  %bf.clear3.i.i = and i16 %write.sroa.5.0.extract.trunc.i, 4095
  %cmp.i9.i.not.i.i = icmp eq i8 %bf.cast.i.i, 0
  br i1 %cmp.i9.i.not.i.i, label %do.body.i.i, label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %if.end.i.i
  switch i8 %switch.masked, label %if.end.i.i.i.unreachabledefault [
    i8 1, label %sw.bb.i.i.i.i
    i8 2, label %sw.bb2.i.i.i.i
    i8 3, label %land.rhs.i.i.i
  ]

sw.bb.i.i.i.i:                                    ; preds = %if.end.i.i.i
  %cmp.i10.i.i.i = icmp eq i16 %47, %bf.clear3.i.i
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i

sw.bb2.i.i.i.i:                                   ; preds = %if.end.i.i.i
  %49 = load i16, ptr %numSms.i.i.i.i, align 4, !tbaa !21
  %50 = udiv i16 %47, %49
  %51 = udiv i16 %bf.clear3.i.i, %49
  %cmp9.i.i.i.i = icmp eq i16 %50, %51
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i

if.end.i.i.i.unreachabledefault:                  ; preds = %if.end.i.i.i
  unreachable

default.unreachable:                              ; preds = %land.rhs.i.i.i, %land.rhs.i.i75.i, %if.end.i.i74.i
  unreachable

_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i: ; preds = %sw.bb2.i.i.i.i, %sw.bb.i.i.i.i
  %retval.0.i.i.i.i = phi i1 [ %cmp9.i.i.i.i, %sw.bb2.i.i.i.i ], [ %cmp.i10.i.i.i, %sw.bb.i.i.i.i ]
  br i1 %retval.0.i.i.i.i, label %land.rhs.i.i.i, label %do.body.i.i

land.rhs.i.i.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i, %if.end.i.i.i
  switch i8 %bf.cast.i.i, label %default.unreachable [
    i8 1, label %sw.bb.i16.i.i.i
    i8 2, label %sw.bb2.i13.i.i.i
    i8 3, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i
  ]

sw.bb.i16.i.i.i:                                  ; preds = %land.rhs.i.i.i
  %cmp.i17.i.i.i = icmp eq i16 %47, %bf.clear3.i.i
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i

sw.bb2.i13.i.i.i:                                 ; preds = %land.rhs.i.i.i
  %52 = load i16, ptr %numSms.i.i.i.i, align 4, !tbaa !21
  %53 = udiv i16 %47, %52
  %54 = udiv i16 %bf.clear3.i.i, %52
  %cmp9.i15.i.i.i = icmp eq i16 %53, %54
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i

_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i: ; preds = %sw.bb2.i13.i.i.i, %sw.bb.i16.i.i.i
  %retval.0.i.i.i = phi i1 [ %cmp.i17.i.i.i, %sw.bb.i16.i.i.i ], [ %cmp9.i15.i.i.i, %sw.bb2.i13.i.i.i ]
  br i1 %retval.0.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i, label %do.body.i.i

do.body.i.i:                                      ; preds = %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i, %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i, %if.end.i.i
  %55 = and i16 %write.sroa.5.0.extract.trunc.i, 16384
  %bf.cast.not.i.i.i.i = icmp eq i16 %55, 0
  br i1 %bf.cast.not.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i, label %if.end.i.i.i.i

if.end.i.i.i.i:                                   ; preds = %do.body.i.i
  %56 = load i16, ptr %numSms.i.i.i.i, align 4, !tbaa !21
  %bf.clear3.i.i.frozen = freeze i16 %bf.clear3.i.i
  %.frozen = freeze i16 %56
  %div16.i.i.i.i.i = udiv i16 %bf.clear3.i.i.frozen, %.frozen
  %57 = mul i16 %div16.i.i.i.i.i, %.frozen
  %rem17.i.i.i.i.i.decomposed = sub i16 %bf.clear3.i.i.frozen, %57
  %58 = load i64, ptr %globalsBase.i.i.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i.i = zext nneg i16 %div16.i.i.i.i.i to i64
  %mul.i.i.i.i.i = shl nuw nsw i64 %conv5.i.i.i.i.i, 30
  %add.i.i.i.i.i = or disjoint i64 %mul.i.i.i.i.i, 39
  %ptr.biased.i.i.i.i.i.i.i = add i64 %add.i.i.i.i.i, %58
  %cond.i.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i.i = load i16, ptr %40, align 8, !tbaa !11
  %globals.val15.i.i.i.i.i = load i16, ptr %41, align 2, !tbaa !15
  %conv.i.i.i.i.i.i = zext i16 %globals.val15.i.i.i.i.i to i64
  %add.i.i.i.i.i.i = add nuw nsw i64 %conv.i.i.i.i.i.i, 1
  %conv1.i.i.i.i.i.i = zext i16 %globals.val.i.i.i.i.i to i64
  %mul.i.i.i.i.i.i = shl nuw nsw i64 %conv1.i.i.i.i.i.i, 1
  %mul3.i.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i.i.i, %add.i.i.i.i.i.i
  %add4.i.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i.i, 32
  %conv7.i.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i, %conv7.i.i.i.i.i
  %add9.i.i.i.i.i = add i64 %mul8.i.i.i.i.i, %cond.i.i.i.i.i.i.i
  %59 = inttoptr i64 %add9.i.i.i.i.i to ptr
  %conv.i.i.i.i.i = and i32 %46, 65535
  %clockBufferHead.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %59, i64 20
  %bf.load.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i, align 4
  %bf.lshr.i.i.i.i.i = lshr i32 %bf.load.i.i.i.i.i, 1
  %cmp3.not.i.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i.i, %conv.i.i.i.i.i
  br i1 %cmp3.not.i.i.i.i.i, label %if.then4.i.i.i.i.i, label %do.end9.i.i.i.i.i

if.then4.i.i.i.i.i:                               ; preds = %if.end.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i, align 4
  %.pre42.i.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i.i, 1
  br label %do.end9.i.i.i.i.i

do.end9.i.i.i.i.i:                                ; preds = %if.then4.i.i.i.i.i, %if.end.i.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i.i, %if.end.i.i.i.i ], [ %.pre42.i.i.i.i.i, %if.then4.i.i.i.i.i ]
  %and.i.i.i.i.i.i = and i64 %add9.i.i.i.i.i, -1073741824
  %60 = inttoptr i64 %and.i.i.i.i.i.i to ptr
  %sub.i.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i.i, %conv.i.i.i.i.i
  %clockBufferSize.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %60, i64 26
  %61 = load i16, ptr %clockBufferSize.i.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i.i = zext i16 %61 to i32
  %cmp17.i.i.i.i.i = icmp slt i32 %sub.i.i.i.i.i, %conv16.i.i.i.i.i
  br i1 %cmp17.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i, label %if.then18.i.i.i.i.i

if.then18.i.i.i.i.i:                              ; preds = %do.end9.i.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i: ; preds = %if.then18.i.i.i.i.i, %do.end9.i.i.i.i.i
  %62 = phi i16 [ %.pre.i.i.i.i.i, %if.then18.i.i.i.i.i ], [ %61, %do.end9.i.i.i.i.i ]
  %63 = urem i16 %write.sroa.0.0.extract.trunc.i, %62
  %rem.i.i.i.i.i = zext i16 %63 to i64
  %vectorClock.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %59, i64 30
  %numThreads.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %60, i64 24
  %64 = load i16, ptr %numThreads.i.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i.i = zext i16 %64 to i64
  %add.ptr.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i.i
  %mul.i8.i.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i.i, %rem.i.i.i.i.i
  %add.ptr.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i.i, i64 %mul.i8.i.i.i.i
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i, %do.body.i.i
  %retval.0.i.i22.i.i = phi ptr [ %add.ptr.i.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i ], [ null, %do.body.i.i ]
  %tobool.not.not.i.i.i = icmp eq ptr %retval.0.i.i22.i.i, null
  br i1 %tobool.not.not.i.i.i, label %cleanup.i.i.i, label %if.then1.i.i.i

if.then1.i.i.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i
  %65 = load i16, ptr %40, align 8, !tbaa !11
  %conv.i.i.i.i = zext i16 %65 to i32
  %cmp.not11.i.i.i.i = icmp eq i16 %65, 0
  br i1 %cmp.not11.i.i.i.i, label %cleanup.i.i.i, label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %if.then1.i.i.i, %for.body.i.i.i.i
  %i.012.i.i.i.i = phi i32 [ %inc.i.i.i.i, %for.body.i.i.i.i ], [ 0, %if.then1.i.i.i ]
  %idxprom.i.i.i.i = zext nneg i32 %i.012.i.i.i.i to i64
  %arrayidx.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idxprom.i.i.i.i
  %66 = load i16, ptr %arrayidx.i.i.i.i, align 2, !tbaa !22
  %arrayidx3.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i.i, i64 %idxprom.i.i.i.i
  %67 = load i16, ptr %arrayidx3.i.i.i.i, align 2, !tbaa !22
  %cmp5.i.i.i.i = icmp uge i16 %66, %67
  %inc.i.i.i.i = add nuw nsw i32 %i.012.i.i.i.i, 1
  %exitcond.not.i.i.i.i = icmp ne i32 %inc.i.i.i.i, %conv.i.i.i.i
  %or.cond.not.i.i.i.i = select i1 %cmp5.i.i.i.i, i1 %exitcond.not.i.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i.i, label %for.body.i.i.i.i, label %cleanup.i.i.i, !llvm.loop !31

cleanup.i.i.i:                                    ; preds = %for.body.i.i.i.i, %if.then1.i.i.i, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i
  %retval.0.i23.i.i = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i ], [ true, %if.then1.i.i.i ], [ %cmp5.i.i.i.i, %for.body.i.i.i.i ]
  br i1 %tobool.not.not.i.i.i, label %cleanup.cont.i.i.i, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i

cleanup.cont.i.i.i:                               ; preds = %cleanup.i.i.i
  %idxprom.i.i.i = zext nneg i16 %bf.clear3.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idxprom.i.i.i
  %68 = load i16, ptr %arrayidx.i.i.i, align 2, !tbaa !22
  %cmp7.i.i.i = icmp uge i16 %68, %write.sroa.0.0.extract.trunc.i
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i: ; preds = %cleanup.cont.i.i.i, %cleanup.i.i.i
  %retval.1.i.i.i = phi i1 [ %retval.0.i23.i.i, %cleanup.i.i.i ], [ %cmp7.i.i.i, %cleanup.cont.i.i.i ]
  br i1 %retval.1.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i, label %if.then9.i.i

if.then9.i.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str1, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i

_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i: ; preds = %if.then9.i.i, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i, %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i, %land.rhs.i.i.i, %for.body.i
  %numReads1.i.i = getelementptr inbounds nuw i8, ptr %45, i64 20
  %69 = load i16, ptr %numReads1.i.i, align 4, !tbaa !32
  %conv.i59.i = zext i16 %69 to i32
  %cmp.not.i.i = icmp eq i16 %69, -1
  br i1 %cmp.not.i.i, label %if.end.i62.i, label %if.then.i60.i

if.then.i60.i:                                    ; preds = %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i
  %inc.i61.i = add nuw i16 %69, 1
  store i16 %inc.i61.i, ptr %numReads1.i.i, align 4, !tbaa !32
  br label %if.end.i62.i

if.end.i62.i:                                     ; preds = %if.then.i60.i, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i
  %70 = load i16, ptr %threadId.i56.i, align 4, !tbaa !22
  %idxprom.i.i64.i = zext i16 %70 to i64
  %arrayidx.i.i65.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idxprom.i.i64.i
  %71 = load i16, ptr %arrayidx.i.i65.i, align 2, !tbaa !22
  %bf.value.i.i.i = and i16 %70, 4095
  %bf.set6.i.i.i = or disjoint i16 %bf.value.i.i.i, %7
  %readClock.sroa.0.0.copyload.i.i = load i16, ptr %45, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.i.i = getelementptr inbounds nuw i8, ptr %45, i64 2
  %readClock.sroa.4.0.copyload.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.i.i, align 2, !tbaa !23
  %bf.clear.i.i = and i16 %readClock.sroa.4.0.copyload.i.i, 4095
  %cmp7.i66.i = icmp ne i16 %bf.clear.i.i, %70
  %cmp9.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.i.i, 0
  %or.cond.not.i.i = select i1 %cmp7.i66.i, i1 %cmp9.i.i, i1 false
  br i1 %or.cond.not.i.i, label %for.cond.i.i, label %if.then10.i.i

for.cond.i.i:                                     ; preds = %if.end.i62.i
  %arrayidx.1.i.i = getelementptr inbounds nuw i8, ptr %45, i64 4
  %readClock.sroa.0.0.copyload.1.i.i = load i16, ptr %arrayidx.1.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.1.i.i = getelementptr inbounds nuw i8, ptr %45, i64 6
  %readClock.sroa.4.0.copyload.1.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.1.i.i, align 2, !tbaa !23
  %bf.clear.1.i.i = and i16 %readClock.sroa.4.0.copyload.1.i.i, 4095
  %cmp7.1.i.i = icmp ne i16 %bf.clear.1.i.i, %70
  %cmp9.1.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.1.i.i, 0
  %or.cond.not.1.i.i = select i1 %cmp7.1.i.i, i1 %cmp9.1.i.i, i1 false
  br i1 %or.cond.not.1.i.i, label %for.cond.1.i.i, label %if.then10.i.i

for.cond.1.i.i:                                   ; preds = %for.cond.i.i
  %arrayidx.2.i.i = getelementptr inbounds nuw i8, ptr %45, i64 8
  %readClock.sroa.0.0.copyload.2.i.i = load i16, ptr %arrayidx.2.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.2.i.i = getelementptr inbounds nuw i8, ptr %45, i64 10
  %readClock.sroa.4.0.copyload.2.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.2.i.i, align 2, !tbaa !23
  %bf.clear.2.i.i = and i16 %readClock.sroa.4.0.copyload.2.i.i, 4095
  %cmp7.2.i.i = icmp ne i16 %bf.clear.2.i.i, %70
  %cmp9.2.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.2.i.i, 0
  %or.cond.not.2.i.i = select i1 %cmp7.2.i.i, i1 %cmp9.2.i.i, i1 false
  br i1 %or.cond.not.2.i.i, label %for.cond.2.i.i, label %if.then10.i.i

for.cond.2.i.i:                                   ; preds = %for.cond.1.i.i
  %arrayidx.3.i.i = getelementptr inbounds nuw i8, ptr %45, i64 12
  %readClock.sroa.0.0.copyload.3.i.i = load i16, ptr %arrayidx.3.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.3.i.i = getelementptr inbounds nuw i8, ptr %45, i64 14
  %readClock.sroa.4.0.copyload.3.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.3.i.i, align 2, !tbaa !23
  %bf.clear.3.i.i = and i16 %readClock.sroa.4.0.copyload.3.i.i, 4095
  %cmp7.3.i.i = icmp ne i16 %bf.clear.3.i.i, %70
  %cmp9.3.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.3.i.i, 0
  %or.cond.not.3.i.i = select i1 %cmp7.3.i.i, i1 %cmp9.3.i.i, i1 false
  br i1 %or.cond.not.3.i.i, label %for.cond.3.i.i, label %if.then10.i.i

for.cond.3.i.i:                                   ; preds = %for.cond.2.i.i
  %72 = atomicrmw add ptr %numReads18.i.i, i32 1 syncscope("block") monotonic, align 8
  %73 = load i32, ptr %rngSeed.i.i, align 16, !tbaa !34
  %conv21.i.i = zext i16 %70 to i32
  %mul.i.i.i.i68.i = mul i32 %72, -862048943
  %or.i.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i.i.i68.i, i32 %mul.i.i.i.i68.i, i32 15)
  %mul1.i.i.i.i.i = mul i32 %or.i.i.i.i.i.i, 461845907
  %xor.i.i.i.i = xor i32 %mul1.i.i.i.i.i, %73
  %or.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i.i.i.i, i32 %xor.i.i.i.i, i32 13)
  %mul.i.i.i.i = mul i32 %or.i.i.i.i.i, 5
  %add.i.i.i.i = add i32 %mul.i.i.i.i, -430675100
  %mul.i.i6.i.i.i = mul i32 %conv21.i.i, -862048943
  %or.i.i.i7.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i6.i.i.i, i32 %mul.i.i6.i.i.i, i32 15)
  %mul1.i.i8.i.i.i = mul i32 %or.i.i.i7.i.i.i, 461845907
  %xor.i9.i.i.i = xor i32 %add.i.i.i.i, %mul1.i.i8.i.i.i
  %or.i.i10.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i9.i.i.i, i32 %xor.i9.i.i.i, i32 13)
  %mul.i11.i.i.i = mul i32 %or.i.i10.i.i.i, 5
  %add.i12.i.i.i = add i32 %mul.i11.i.i.i, -430675100
  %shr.i.i.i.i = lshr i32 %add.i12.i.i.i, 16
  %74 = xor i32 %add.i12.i.i.i, %shr.i.i.i.i
  %xor.i13.i.i.i = xor i32 %74, 8
  %mul.i14.i.i.i = mul i32 %xor.i13.i.i.i, -2048144789
  %shr1.i.i.i.i = lshr i32 %mul.i14.i.i.i, 13
  %xor2.i.i.i.i = xor i32 %shr1.i.i.i.i, %mul.i14.i.i.i
  %mul3.i.i.i.i = mul i32 %xor2.i.i.i.i, -1028477387
  %shr4.i.i.i.i = lshr i32 %mul3.i.i.i.i, 16
  %xor5.i.i.i.i = xor i32 %shr4.i.i.i.i, %mul3.i.i.i.i
  %rem.i.i = urem i32 %xor5.i.i.i.i, %conv.i59.i
  %cmp24.i.i = icmp samesign ult i32 %rem.i.i, 4
  br i1 %cmp24.i.i, label %if.then25.i.i, label %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i

if.then10.i.i:                                    ; preds = %for.cond.2.i.i, %for.cond.1.i.i, %for.cond.i.i, %if.end.i62.i
  %arrayidx.lcssa.i.i = phi ptr [ %45, %if.end.i62.i ], [ %arrayidx.1.i.i, %for.cond.i.i ], [ %arrayidx.2.i.i, %for.cond.1.i.i ], [ %arrayidx.3.i.i, %for.cond.2.i.i ]
  %readClock.sroa.4.0.arrayidx.sroa_idx.le.i.i = getelementptr inbounds nuw i8, ptr %arrayidx.lcssa.i.i, i64 2
  store i16 %71, ptr %arrayidx.lcssa.i.i, align 4, !tbaa !22
  store i16 %bf.set6.i.i.i, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.le.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i

if.then25.i.i:                                    ; preds = %for.cond.3.i.i
  %idxprom27.i.i = zext nneg i32 %rem.i.i to i64
  %arrayidx28.i.i = getelementptr inbounds nuw [4 x i8], ptr %45, i64 %idxprom27.i.i
  %scalarClock.sroa.5.0.arrayidx28.sroa_idx.i.i = getelementptr inbounds nuw i8, ptr %arrayidx28.i.i, i64 2
  store i16 %71, ptr %arrayidx28.i.i, align 4, !tbaa !22
  store i16 %bf.set6.i.i.i, ptr %scalarClock.sroa.5.0.arrayidx28.sroa_idx.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i

_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i: ; preds = %if.then25.i.i, %if.then10.i.i, %for.cond.3.i.i
  %inc.i = add nuw i8 %i.0114.i, 1
  %cmp6.i = icmp ult i8 %inc.i, %38
  br i1 %cmp6.i, label %for.body.i, label %for.cond.cleanup.i, !llvm.loop !52

for.body17.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, %for.body17.lr.ph.i
  %i10.0116.i = phi i8 [ 0, %for.body17.lr.ph.i ], [ %inc25.i, %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i ]
  %idxprom20.i = zext i8 %i10.0116.i to i64
  %arrayidx21.i = getelementptr inbounds nuw [8 x i8], ptr %cells.i.i, i64 %idxprom20.i
  %75 = load ptr, ptr %arrayidx21.i, align 8, !tbaa !49
  %writeClock22.i = getelementptr inbounds nuw i8, ptr %75, i64 16
  %76 = load i32, ptr %writeClock22.i, align 4
  %write18.sroa.0.0.extract.trunc.i = trunc i32 %76 to i16
  %write18.sroa.4.0.extract.shift.i = lshr i32 %76, 16
  %write18.sroa.4.0.extract.trunc.i = trunc nuw i32 %write18.sroa.4.0.extract.shift.i to i16
  %77 = and i16 %write18.sroa.4.0.extract.trunc.i, 16384
  %bf.cast.not.i.i = icmp eq i16 %77, 0
  br i1 %bf.cast.not.i.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, label %if.end.i70.i

if.end.i70.i:                                     ; preds = %for.body17.i
  %78 = load i16, ptr %threadId.i71.i, align 4, !tbaa !22
  %bf.lshr2.i.i = lshr i16 %write18.sroa.4.0.extract.trunc.i, 12
  %79 = trunc nuw nsw i16 %bf.lshr2.i.i to i8
  %bf.cast4.i.i = and i8 %79, 3
  %bf.clear7.i.i = and i16 %write18.sroa.4.0.extract.trunc.i, 4095
  %cmp.i9.i.not.i73.i = icmp eq i8 %bf.cast4.i.i, 0
  br i1 %cmp.i9.i.not.i73.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, label %if.end.i.i74.i

if.end.i.i74.i:                                   ; preds = %if.end.i70.i
  switch i8 %switch.masked, label %default.unreachable [
    i8 1, label %sw.bb.i.i.i109.i
    i8 2, label %sw.bb2.i.i.i104.i
    i8 3, label %land.rhs.i.i75.i
  ]

sw.bb.i.i.i109.i:                                 ; preds = %if.end.i.i74.i
  %cmp.i10.i.i110.i = icmp eq i16 %78, %bf.clear7.i.i
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i

sw.bb2.i.i.i104.i:                                ; preds = %if.end.i.i74.i
  %80 = load i16, ptr %numSms.i.i.i105.i, align 4, !tbaa !21
  %81 = udiv i16 %78, %80
  %82 = udiv i16 %bf.clear7.i.i, %80
  %cmp9.i.i.i106.i = icmp eq i16 %81, %82
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i

_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i: ; preds = %sw.bb2.i.i.i104.i, %sw.bb.i.i.i109.i
  %retval.0.i.i.i108.i = phi i1 [ %cmp9.i.i.i106.i, %sw.bb2.i.i.i104.i ], [ %cmp.i10.i.i110.i, %sw.bb.i.i.i109.i ]
  br i1 %retval.0.i.i.i108.i, label %land.rhs.i.i75.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

land.rhs.i.i75.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i, %if.end.i.i74.i
  switch i8 %bf.cast4.i.i, label %default.unreachable [
    i8 1, label %sw.bb.i16.i.i101.i
    i8 2, label %sw.bb2.i13.i.i96.i
    i8 3, label %if.end10.i.i
  ]

sw.bb.i16.i.i101.i:                               ; preds = %land.rhs.i.i75.i
  %cmp.i17.i.i102.i = icmp eq i16 %78, %bf.clear7.i.i
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i

sw.bb2.i13.i.i96.i:                               ; preds = %land.rhs.i.i75.i
  %83 = load i16, ptr %numSms.i.i.i105.i, align 4, !tbaa !21
  %84 = udiv i16 %78, %83
  %85 = udiv i16 %bf.clear7.i.i, %83
  %cmp9.i15.i.i98.i = icmp eq i16 %84, %85
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i

_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i: ; preds = %sw.bb2.i13.i.i96.i, %sw.bb.i16.i.i101.i
  %retval.0.i.i100.i = phi i1 [ %cmp.i17.i.i102.i, %sw.bb.i16.i.i101.i ], [ %cmp9.i15.i.i98.i, %sw.bb2.i13.i.i96.i ]
  br i1 %retval.0.i.i100.i, label %if.end10.i.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

if.end10.i.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i, %land.rhs.i.i75.i
  %86 = load i16, ptr %numSms.i.i.i105.i, align 4, !tbaa !21
  %bf.clear7.i.i.frozen = freeze i16 %bf.clear7.i.i
  %.frozen36 = freeze i16 %86
  %div16.i.i.i.i = udiv i16 %bf.clear7.i.i.frozen, %.frozen36
  %87 = mul i16 %div16.i.i.i.i, %.frozen36
  %rem17.i.i.i.i.decomposed = sub i16 %bf.clear7.i.i.frozen, %87
  %88 = load i64, ptr %globalsBase.i.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i = zext nneg i16 %div16.i.i.i.i to i64
  %mul.i.i.i79.i = shl nuw nsw i64 %conv5.i.i.i.i, 30
  %add.i.i.i80.i = or disjoint i64 %mul.i.i.i79.i, 39
  %ptr.biased.i.i.i.i.i.i = add i64 %add.i.i.i80.i, %88
  %cond.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i = load i16, ptr %43, align 8, !tbaa !11
  %globals.val15.i.i.i.i = load i16, ptr %44, align 2, !tbaa !15
  %conv.i.i.i.i81.i = zext i16 %globals.val15.i.i.i.i to i64
  %add.i.i.i.i82.i = add nuw nsw i64 %conv.i.i.i.i81.i, 1
  %conv1.i.i.i.i.i = zext i16 %globals.val.i.i.i.i to i64
  %mul.i.i.i.i83.i = shl nuw nsw i64 %conv1.i.i.i.i.i, 1
  %mul3.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i83.i, %add.i.i.i.i82.i
  %add4.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i, 32
  %conv7.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i, %conv7.i.i.i.i
  %add9.i.i.i.i = add i64 %mul8.i.i.i.i, %cond.i.i.i.i.i.i
  %89 = inttoptr i64 %add9.i.i.i.i to ptr
  %conv.i.i.i84.i = and i32 %76, 65535
  %cmp.not.i.i.i.i = icmp eq i16 %write18.sroa.0.0.extract.trunc.i, 0
  br i1 %cmp.not.i.i.i.i, label %if.then.i.i.i.i, label %do.body1.i.i.i.i

if.then.i.i.i.i:                                  ; preds = %if.end10.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str3, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.body1.i.i.i.i

do.body1.i.i.i.i:                                 ; preds = %if.then.i.i.i.i, %if.end10.i.i
  %clockBufferHead.i.i.i.i = getelementptr inbounds nuw i8, ptr %89, i64 20
  %bf.load.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i, align 4
  %bf.lshr.i.i.i.i = lshr i32 %bf.load.i.i.i.i, 1
  %cmp3.not.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i, %conv.i.i.i84.i
  br i1 %cmp3.not.i.i.i.i, label %if.then4.i.i.i.i, label %do.end9.i.i.i.i

if.then4.i.i.i.i:                                 ; preds = %do.body1.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i, align 4
  %.pre42.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i, 1
  br label %do.end9.i.i.i.i

do.end9.i.i.i.i:                                  ; preds = %if.then4.i.i.i.i, %do.body1.i.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i, %do.body1.i.i.i.i ], [ %.pre42.i.i.i.i, %if.then4.i.i.i.i ]
  %and.i.i.i.i85.i = and i64 %add9.i.i.i.i, -1073741824
  %90 = inttoptr i64 %and.i.i.i.i85.i to ptr
  %sub.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i, %conv.i.i.i84.i
  %clockBufferSize.i.i.i.i = getelementptr inbounds nuw i8, ptr %90, i64 26
  %91 = load i16, ptr %clockBufferSize.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i = zext i16 %91 to i32
  %cmp17.i.i.i.i = icmp slt i32 %sub.i.i.i.i, %conv16.i.i.i.i
  br i1 %cmp17.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, label %if.then18.i.i.i.i

if.then18.i.i.i.i:                                ; preds = %do.end9.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i: ; preds = %if.then18.i.i.i.i, %do.end9.i.i.i.i
  %92 = phi i16 [ %.pre.i.i.i.i, %if.then18.i.i.i.i ], [ %91, %do.end9.i.i.i.i ]
  %93 = urem i16 %write18.sroa.0.0.extract.trunc.i, %92
  %rem.i.i.i.i = zext i16 %93 to i64
  %vectorClock.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %89, i64 30
  %numThreads.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %90, i64 24
  %94 = load i16, ptr %numThreads.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i = zext i16 %94 to i64
  %add.ptr.i.i.i.i86.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i
  %mul.i8.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i, %rem.i.i.i.i
  %add.ptr.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i86.i, i64 %mul.i8.i.i.i
  %95 = load i16, ptr %43, align 8, !tbaa !11
  %cmp2.not.i.i = icmp eq i16 %95, 0
  br i1 %cmp2.not.i.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, label %for.body.i87.i

for.cond.cleanup.i93.i:                           ; preds = %for.inc.i.i
  br i1 %changed.1.off0.i.i, label %if.then25.i94.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

for.body.i87.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, %for.inc.i.i
  %96 = phi i16 [ %99, %for.inc.i.i ], [ %95, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ]
  %i.04.i.i = phi i32 [ %inc.i90.i, %for.inc.i.i ], [ 0, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ]
  %changed.0.off03.i.i = phi i1 [ %changed.1.off0.i.i, %for.inc.i.i ], [ false, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ]
  %idxprom.i88.i = zext nneg i32 %i.04.i.i to i64
  %arrayidx.i89.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i, i64 %idxprom.i88.i
  %97 = load i16, ptr %arrayidx.i89.i, align 2, !tbaa !22
  %arrayidx15.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i, i64 %idxprom.i88.i
  %98 = load i16, ptr %arrayidx15.i.i, align 2, !tbaa !22
  %cmp17.i.i = icmp ult i16 %97, %98
  br i1 %cmp17.i.i, label %if.then18.i.i, label %for.inc.i.i

if.then18.i.i:                                    ; preds = %for.body.i87.i
  store i16 %98, ptr %arrayidx.i89.i, align 2, !tbaa !22
  %.pre.i.i = load i16, ptr %43, align 8, !tbaa !11
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %if.then18.i.i, %for.body.i87.i
  %99 = phi i16 [ %.pre.i.i, %if.then18.i.i ], [ %96, %for.body.i87.i ]
  %changed.1.off0.i.i = phi i1 [ true, %if.then18.i.i ], [ %changed.0.off03.i.i, %for.body.i87.i ]
  %inc.i90.i = add nuw nsw i32 %i.04.i.i, 1
  %conv.i91.i = zext i16 %99 to i32
  %cmp.i92.i = icmp samesign ult i32 %inc.i90.i, %conv.i91.i
  br i1 %cmp.i92.i, label %for.body.i87.i, label %for.cond.cleanup.i93.i, !llvm.loop !53

if.then25.i94.i:                                  ; preds = %for.cond.cleanup.i93.i
  %bf.load26.i.i = load i32, ptr %clockBufferDirty.i95.i, align 4
  %bf.set.i.i = or i32 %bf.load26.i.i, 1
  store i32 %bf.set.i.i, ptr %clockBufferDirty.i95.i, align 4
  br label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i: ; preds = %if.then25.i94.i, %for.cond.cleanup.i93.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i, %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i, %if.end.i70.i, %for.body17.i
  %inc25.i = add nuw i8 %i10.0116.i, 1
  %cmp15.i = icmp ult i8 %inc25.i, %38
  br i1 %cmp15.i, label %for.body17.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, !llvm.loop !54

_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit: ; preds = %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i, %for.cond.cleanup.i, %for.cond11.preheader.i
  store ptr %file, ptr %agg.tmp5, align 8, !tbaa !55
  store i32 %line, ptr %loc.sroa.5.0.agg.tmp5.sroa_idx, align 8, !tbaa !3
  call fastcc void @_ZN4gsan12_GLOBAL__N_115endAtomicAccessEPNS_16AtomicEventStateEbbjjNS_8LocationE(ptr noundef nonnull %event, i1 noundef zeroext true, i1 noundef zeroext true, i32 noundef %sem, i32 noundef %scope, ptr noundef nonnull byval(%"struct.gsan::Location") align 8 %agg.tmp5) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %event) #9
  br label %for.inc

for.inc:                                          ; preds = %for.body, %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit
  %inc = add nuw nsw i32 %i.021, 1
  %exitcond.not = icmp eq i32 %inc, %numElems
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !56
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define internal fastcc void @_ZN4gsan12_GLOBAL__N_115endAtomicAccessEPNS_16AtomicEventStateEbbjjNS_8LocationE(ptr noundef captures(none) %event, i1 noundef zeroext %pred, i1 noundef zeroext %didWrite, i32 noundef %semRaw, i32 noundef %scopeRaw, ptr noundef readonly byval(%"struct.gsan::Location") align 8 captures(none) %loc) unnamed_addr #1 {
entry:
  br i1 %pred, label %lor.lhs.false, label %return

lor.lhs.false:                                    ; preds = %entry
  %0 = load ptr, ptr %event, align 8, !tbaa !45
  %cmp = icmp eq ptr %0, null
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %lor.lhs.false
  %switch.tableidx.i = add i32 %semRaw, -1
  %1 = icmp ult i32 %switch.tableidx.i, 4
  br i1 %1, label %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit, label %sw.default.i

sw.default.i:                                     ; preds = %if.end
  tail call void @llvm.trap()
  unreachable

_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit:   ; preds = %if.end
  %switch.tableidx = add i32 %scopeRaw, -1
  %2 = icmp ult i32 %switch.tableidx, 3
  br i1 %2, label %switch.lookup, label %sw.default.i87

sw.default.i87:                                   ; preds = %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit
  tail call void @llvm.trap()
  unreachable

switch.lookup:                                    ; preds = %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit
  %switch.cast = trunc nuw i32 %switch.tableidx to i24
  %switch.shiftamt = shl nuw nsw i24 %switch.cast, 3
  %switch.downshift = lshr i24 196866, %switch.shiftamt
  %switch.masked = trunc i24 %switch.downshift to i8
  br i1 %didWrite, label %for.cond.preheader, label %if.end56

for.cond.preheader:                               ; preds = %switch.lookup
  %numCells = getelementptr inbounds nuw i8, ptr %event, i64 32
  %3 = load i8, ptr %numCells, align 8, !tbaa !48
  %cmp7305.not = icmp eq i8 %3, 0
  br i1 %cmp7305.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %cells = getelementptr inbounds nuw i8, ptr %event, i64 8
  %agg.tmp.sroa.0.0.copyload = load ptr, ptr %loc, align 8, !tbaa !55
  %agg.tmp.sroa.2.0.loc.sroa_idx = getelementptr inbounds nuw i8, ptr %loc, i64 8
  %agg.tmp.sroa.2.0.copyload = load i32, ptr %agg.tmp.sroa.2.0.loc.sroa_idx, align 8, !tbaa !3
  %threadId.i92 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %4 = ptrtoint ptr %0 to i64
  %and.i.i98 = and i64 %4, -1073741824
  %5 = inttoptr i64 %and.i.i98 to ptr
  %numSms.i.i.i200 = getelementptr inbounds nuw i8, ptr %5, i64 20
  %globalsBase.i.i.i.i118 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %6 = getelementptr i8, ptr %5, i64 24
  %7 = getelementptr i8, ptr %5, i64 26
  %cmp.i37.i.i.i.i192 = icmp eq ptr %agg.tmp.sroa.0.0.copyload, null
  %cond.i38.i.i.i.i193 = select i1 %cmp.i37.i.i.i.i192, ptr @.str, ptr %agg.tmp.sroa.0.0.copyload
  %vectorClock.i.i.i164 = getelementptr inbounds nuw i8, ptr %0, i64 30
  br label %for.body

for.cond.cleanup:                                 ; preds = %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit, %for.cond.preheader
  %.not = icmp ult i32 %switch.tableidx.i, 2
  br i1 %.not, label %if.else, label %if.then19

for.body:                                         ; preds = %for.body.lr.ph, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit
  %i.0306 = phi i8 [ 0, %for.body.lr.ph ], [ %inc16, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit ]
  %idxprom = zext i8 %i.0306 to i64
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr %cells, i64 %idxprom
  %8 = load ptr, ptr %arrayidx, align 8, !tbaa !49
  br label %for.body11

for.cond.cleanup10:                               ; preds = %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206
  %writeClock = getelementptr inbounds nuw i8, ptr %8, i64 16
  %9 = load i16, ptr %writeClock, align 4, !tbaa !27
  %cmp.i = icmp eq i16 %9, 0
  br i1 %cmp.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit, label %if.end.i

if.end.i:                                         ; preds = %for.cond.cleanup10
  %10 = load i16, ptr %threadId.i92, align 4, !tbaa !22
  %scope.i = getelementptr inbounds nuw i8, ptr %8, i64 18
  %bf.load.i = load i16, ptr %scope.i, align 2
  %bf.lshr.i = lshr i16 %bf.load.i, 12
  %11 = trunc nuw nsw i16 %bf.lshr.i to i8
  %bf.cast.i = and i8 %11, 3
  %bf.clear3.i = and i16 %bf.load.i, 4095
  %cmp.i9.i.not.i = icmp eq i8 %bf.cast.i, 0
  br i1 %cmp.i9.i.not.i, label %do.body.i, label %if.end.i.i

if.end.i.i:                                       ; preds = %if.end.i
  switch i8 %switch.masked, label %if.end.i.i.unreachabledefault [
    i8 1, label %sw.bb.i.i.i
    i8 2, label %sw.bb2.i.i.i
    i8 3, label %land.rhs.i.i
  ]

sw.bb.i.i.i:                                      ; preds = %if.end.i.i
  %cmp.i10.i.i = icmp eq i16 %10, %bf.clear3.i
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i

sw.bb2.i.i.i:                                     ; preds = %if.end.i.i
  %12 = load i16, ptr %numSms.i.i.i200, align 4, !tbaa !21
  %13 = udiv i16 %10, %12
  %14 = udiv i16 %bf.clear3.i, %12
  %cmp9.i.i.i = icmp eq i16 %13, %14
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i

if.end.i.i.unreachabledefault:                    ; preds = %if.end.i.i
  unreachable

default.unreachable:                              ; preds = %land.rhs.i.i, %land.rhs.i.i101, %if.end.i.i100
  unreachable

_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i: ; preds = %sw.bb2.i.i.i, %sw.bb.i.i.i
  %retval.0.i.i.i = phi i1 [ %cmp9.i.i.i, %sw.bb2.i.i.i ], [ %cmp.i10.i.i, %sw.bb.i.i.i ]
  br i1 %retval.0.i.i.i, label %land.rhs.i.i, label %do.body.i

land.rhs.i.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i, %if.end.i.i
  switch i8 %bf.cast.i, label %default.unreachable [
    i8 1, label %sw.bb.i16.i.i
    i8 2, label %sw.bb2.i13.i.i
    i8 3, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit
  ]

sw.bb.i16.i.i:                                    ; preds = %land.rhs.i.i
  %cmp.i17.i.i = icmp eq i16 %10, %bf.clear3.i
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i

sw.bb2.i13.i.i:                                   ; preds = %land.rhs.i.i
  %15 = load i16, ptr %numSms.i.i.i200, align 4, !tbaa !21
  %16 = udiv i16 %10, %15
  %17 = udiv i16 %bf.clear3.i, %15
  %cmp9.i15.i.i = icmp eq i16 %16, %17
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i

_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i: ; preds = %sw.bb2.i13.i.i, %sw.bb.i16.i.i
  %retval.0.i.i = phi i1 [ %cmp.i17.i.i, %sw.bb.i16.i.i ], [ %cmp9.i15.i.i, %sw.bb2.i13.i.i ]
  br i1 %retval.0.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit, label %do.body.i

do.body.i:                                        ; preds = %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i, %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i, %if.end.i
  %18 = and i16 %bf.load.i, 16384
  %bf.cast.not.i.i.i = icmp eq i16 %18, 0
  br i1 %bf.cast.not.i.i.i, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i, label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %do.body.i
  %19 = load i16, ptr %numSms.i.i.i200, align 4, !tbaa !21
  %bf.clear3.i.frozen = freeze i16 %bf.clear3.i
  %.frozen = freeze i16 %19
  %div16.i.i.i.i = udiv i16 %bf.clear3.i.frozen, %.frozen
  %20 = mul i16 %div16.i.i.i.i, %.frozen
  %rem17.i.i.i.i.decomposed = sub i16 %bf.clear3.i.frozen, %20
  %21 = load i64, ptr %globalsBase.i.i.i.i118, align 8, !tbaa !20
  %conv5.i.i.i.i = zext nneg i16 %div16.i.i.i.i to i64
  %mul.i.i.i.i = shl nuw nsw i64 %conv5.i.i.i.i, 30
  %add.i.i.i.i = or disjoint i64 %mul.i.i.i.i, 39
  %ptr.biased.i.i.i.i.i.i = add i64 %add.i.i.i.i, %21
  %cond.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i = load i16, ptr %6, align 8, !tbaa !11
  %globals.val15.i.i.i.i = load i16, ptr %7, align 2, !tbaa !15
  %conv.i.i.i.i.i = zext i16 %globals.val15.i.i.i.i to i64
  %add.i.i.i.i.i = add nuw nsw i64 %conv.i.i.i.i.i, 1
  %conv1.i.i.i.i.i = zext i16 %globals.val.i.i.i.i to i64
  %mul.i.i.i.i.i = shl nuw nsw i64 %conv1.i.i.i.i.i, 1
  %mul3.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i.i, %add.i.i.i.i.i
  %add4.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i, 32
  %conv7.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i, %conv7.i.i.i.i
  %add9.i.i.i.i = add i64 %mul8.i.i.i.i, %cond.i.i.i.i.i.i
  %22 = inttoptr i64 %add9.i.i.i.i to ptr
  %conv.i.i.i.i = zext i16 %9 to i32
  %clockBufferHead.i.i.i.i = getelementptr inbounds nuw i8, ptr %22, i64 20
  %bf.load.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i, align 4
  %bf.lshr.i.i.i.i = lshr i32 %bf.load.i.i.i.i, 1
  %cmp3.not.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i, %conv.i.i.i.i
  br i1 %cmp3.not.i.i.i.i, label %if.then4.i.i.i.i, label %do.end9.i.i.i.i

if.then4.i.i.i.i:                                 ; preds = %if.end.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i.i193, i32 noundef %agg.tmp.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i, align 4
  %.pre42.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i, 1
  br label %do.end9.i.i.i.i

do.end9.i.i.i.i:                                  ; preds = %if.then4.i.i.i.i, %if.end.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i, %if.end.i.i.i ], [ %.pre42.i.i.i.i, %if.then4.i.i.i.i ]
  %and.i.i.i.i.i = and i64 %add9.i.i.i.i, -1073741824
  %23 = inttoptr i64 %and.i.i.i.i.i to ptr
  %sub.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i, %conv.i.i.i.i
  %clockBufferSize.i.i.i.i = getelementptr inbounds nuw i8, ptr %23, i64 26
  %24 = load i16, ptr %clockBufferSize.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i = zext i16 %24 to i32
  %cmp17.i.i.i.i = icmp slt i32 %sub.i.i.i.i, %conv16.i.i.i.i
  br i1 %cmp17.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, label %if.then18.i.i.i.i

if.then18.i.i.i.i:                                ; preds = %do.end9.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i38.i.i.i.i193, i32 noundef %agg.tmp.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i: ; preds = %if.then18.i.i.i.i, %do.end9.i.i.i.i
  %25 = phi i16 [ %.pre.i.i.i.i, %if.then18.i.i.i.i ], [ %24, %do.end9.i.i.i.i ]
  %26 = urem i16 %9, %25
  %rem.i.i.i.i = zext i16 %26 to i64
  %vectorClock.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %22, i64 30
  %numThreads.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %23, i64 24
  %27 = load i16, ptr %numThreads.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i = zext i16 %27 to i64
  %add.ptr.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i
  %mul.i8.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i, %rem.i.i.i.i
  %add.ptr.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i, i64 %mul.i8.i.i.i
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, %do.body.i
  %retval.0.i.i22.i = phi ptr [ %add.ptr.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ], [ null, %do.body.i ]
  %tobool.not.not.i.i = icmp eq ptr %retval.0.i.i22.i, null
  br i1 %tobool.not.not.i.i, label %cleanup.i.i, label %if.then1.i.i

if.then1.i.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i
  %28 = load i16, ptr %6, align 8, !tbaa !11
  %conv.i.i.i = zext i16 %28 to i32
  %cmp.not11.i.i.i = icmp eq i16 %28, 0
  br i1 %cmp.not11.i.i.i, label %cleanup.i.i, label %for.body.i.i.i

for.body.i.i.i:                                   ; preds = %if.then1.i.i, %for.body.i.i.i
  %i.012.i.i.i = phi i32 [ %inc.i.i.i, %for.body.i.i.i ], [ 0, %if.then1.i.i ]
  %idxprom.i.i.i = zext nneg i32 %i.012.i.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i164, i64 %idxprom.i.i.i
  %29 = load i16, ptr %arrayidx.i.i.i, align 2, !tbaa !22
  %arrayidx3.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i, i64 %idxprom.i.i.i
  %30 = load i16, ptr %arrayidx3.i.i.i, align 2, !tbaa !22
  %cmp5.i.i.i = icmp uge i16 %29, %30
  %inc.i.i.i = add nuw nsw i32 %i.012.i.i.i, 1
  %exitcond.not.i.i.i = icmp ne i32 %inc.i.i.i, %conv.i.i.i
  %or.cond.not.i.i.i = select i1 %cmp5.i.i.i, i1 %exitcond.not.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i, label %for.body.i.i.i, label %cleanup.i.i, !llvm.loop !31

cleanup.i.i:                                      ; preds = %for.body.i.i.i, %if.then1.i.i, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i
  %retval.0.i23.i = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i ], [ true, %if.then1.i.i ], [ %cmp5.i.i.i, %for.body.i.i.i ]
  br i1 %tobool.not.not.i.i, label %cleanup.cont.i.i, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i

cleanup.cont.i.i:                                 ; preds = %cleanup.i.i
  %bf.load.i.i = load i16, ptr %scope.i, align 2
  %bf.clear.i.i = and i16 %bf.load.i.i, 4095
  %idxprom.i.i = zext nneg i16 %bf.clear.i.i to i64
  %arrayidx.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i164, i64 %idxprom.i.i
  %31 = load i16, ptr %arrayidx.i.i, align 2, !tbaa !22
  %32 = load i16, ptr %writeClock, align 4, !tbaa !27
  %cmp7.i.i = icmp uge i16 %31, %32
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i: ; preds = %cleanup.cont.i.i, %cleanup.i.i
  %retval.1.i.i = phi i1 [ %retval.0.i23.i, %cleanup.i.i ], [ %cmp7.i.i, %cleanup.cont.i.i ]
  br i1 %retval.1.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit, label %if.then9.i

if.then9.i:                                       ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i
  tail call void @__assertfail(ptr noundef nonnull @.str8, ptr noundef nonnull %cond.i38.i.i.i.i193, i32 noundef %agg.tmp.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit

_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit: ; preds = %for.cond.cleanup10, %land.rhs.i.i, %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i, %if.then9.i
  %inc16 = add nuw i8 %i.0306, 1
  %33 = load i8, ptr %numCells, align 8, !tbaa !48
  %cmp7 = icmp ult i8 %inc16, %33
  br i1 %cmp7, label %for.body, label %for.cond.cleanup, !llvm.loop !57

for.body11:                                       ; preds = %for.body, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206
  %iRead.0304 = phi i32 [ 0, %for.body ], [ %inc, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206 ]
  %idxprom12 = zext nneg i32 %iRead.0304 to i64
  %arrayidx13 = getelementptr inbounds nuw [4 x i8], ptr %8, i64 %idxprom12
  %34 = load i16, ptr %arrayidx13, align 4, !tbaa !27
  %cmp.i90 = icmp eq i16 %34, 0
  br i1 %cmp.i90, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206, label %if.end.i91

if.end.i91:                                       ; preds = %for.body11
  %35 = load i16, ptr %threadId.i92, align 4, !tbaa !22
  %scope.i93 = getelementptr inbounds nuw i8, ptr %arrayidx13, i64 2
  %bf.load.i94 = load i16, ptr %scope.i93, align 2
  %bf.lshr.i95 = lshr i16 %bf.load.i94, 12
  %36 = trunc nuw nsw i16 %bf.lshr.i95 to i8
  %bf.cast.i96 = and i8 %36, 3
  %bf.clear3.i97 = and i16 %bf.load.i94, 4095
  %cmp.i9.i.not.i99 = icmp eq i8 %bf.cast.i96, 0
  br i1 %cmp.i9.i.not.i99, label %do.body.i107, label %if.end.i.i100

if.end.i.i100:                                    ; preds = %if.end.i91
  switch i8 %switch.masked, label %default.unreachable [
    i8 1, label %sw.bb.i.i.i204
    i8 2, label %sw.bb2.i.i.i199
    i8 3, label %land.rhs.i.i101
  ]

sw.bb.i.i.i204:                                   ; preds = %if.end.i.i100
  %cmp.i10.i.i205 = icmp eq i16 %35, %bf.clear3.i97
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i202

sw.bb2.i.i.i199:                                  ; preds = %if.end.i.i100
  %37 = load i16, ptr %numSms.i.i.i200, align 4, !tbaa !21
  %38 = udiv i16 %35, %37
  %39 = udiv i16 %bf.clear3.i97, %37
  %cmp9.i.i.i201 = icmp eq i16 %38, %39
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i202

_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i202: ; preds = %sw.bb2.i.i.i199, %sw.bb.i.i.i204
  %retval.0.i.i.i203 = phi i1 [ %cmp9.i.i.i201, %sw.bb2.i.i.i199 ], [ %cmp.i10.i.i205, %sw.bb.i.i.i204 ]
  br i1 %retval.0.i.i.i203, label %land.rhs.i.i101, label %do.body.i107

land.rhs.i.i101:                                  ; preds = %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i202, %if.end.i.i100
  switch i8 %bf.cast.i96, label %default.unreachable [
    i8 1, label %sw.bb.i16.i.i196
    i8 2, label %sw.bb2.i13.i.i102
    i8 3, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206
  ]

sw.bb.i16.i.i196:                                 ; preds = %land.rhs.i.i101
  %cmp.i17.i.i197 = icmp eq i16 %35, %bf.clear3.i97
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i105

sw.bb2.i13.i.i102:                                ; preds = %land.rhs.i.i101
  %40 = load i16, ptr %numSms.i.i.i200, align 4, !tbaa !21
  %41 = udiv i16 %35, %40
  %42 = udiv i16 %bf.clear3.i97, %40
  %cmp9.i15.i.i104 = icmp eq i16 %41, %42
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i105

_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i105: ; preds = %sw.bb2.i13.i.i102, %sw.bb.i16.i.i196
  %retval.0.i.i106 = phi i1 [ %cmp.i17.i.i197, %sw.bb.i16.i.i196 ], [ %cmp9.i15.i.i104, %sw.bb2.i13.i.i102 ]
  br i1 %retval.0.i.i106, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206, label %do.body.i107

do.body.i107:                                     ; preds = %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i105, %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i202, %if.end.i91
  %43 = and i16 %bf.load.i94, 16384
  %bf.cast.not.i.i.i111 = icmp eq i16 %43, 0
  br i1 %bf.cast.not.i.i.i111, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i159, label %if.end.i.i.i112

if.end.i.i.i112:                                  ; preds = %do.body.i107
  %44 = load i16, ptr %numSms.i.i.i200, align 4, !tbaa !21
  %bf.clear3.i97.frozen = freeze i16 %bf.clear3.i97
  %.frozen316 = freeze i16 %44
  %div16.i.i.i.i116 = udiv i16 %bf.clear3.i97.frozen, %.frozen316
  %45 = mul i16 %div16.i.i.i.i116, %.frozen316
  %rem17.i.i.i.i117.decomposed = sub i16 %bf.clear3.i97.frozen, %45
  %46 = load i64, ptr %globalsBase.i.i.i.i118, align 8, !tbaa !20
  %conv5.i.i.i.i119 = zext nneg i16 %div16.i.i.i.i116 to i64
  %mul.i.i.i.i120 = shl nuw nsw i64 %conv5.i.i.i.i119, 30
  %add.i.i.i.i121 = or disjoint i64 %mul.i.i.i.i120, 39
  %ptr.biased.i.i.i.i.i.i122 = add i64 %add.i.i.i.i121, %46
  %cond.i.i.i.i.i.i123 = and i64 %ptr.biased.i.i.i.i.i.i122, -8
  %globals.val.i.i.i.i124 = load i16, ptr %6, align 8, !tbaa !11
  %globals.val15.i.i.i.i125 = load i16, ptr %7, align 2, !tbaa !15
  %conv.i.i.i.i.i126 = zext i16 %globals.val15.i.i.i.i125 to i64
  %add.i.i.i.i.i127 = add nuw nsw i64 %conv.i.i.i.i.i126, 1
  %conv1.i.i.i.i.i128 = zext i16 %globals.val.i.i.i.i124 to i64
  %mul.i.i.i.i.i129 = shl nuw nsw i64 %conv1.i.i.i.i.i128, 1
  %mul3.i.i.i.i.i130 = mul nuw nsw i64 %mul.i.i.i.i.i129, %add.i.i.i.i.i127
  %add4.i.i.i.i.i131 = add nuw nsw i64 %mul3.i.i.i.i.i130, 32
  %conv7.i.i.i.i132 = zext nneg i16 %rem17.i.i.i.i117.decomposed to i64
  %mul8.i.i.i.i133 = mul nuw nsw i64 %add4.i.i.i.i.i131, %conv7.i.i.i.i132
  %add9.i.i.i.i134 = add i64 %mul8.i.i.i.i133, %cond.i.i.i.i.i.i123
  %47 = inttoptr i64 %add9.i.i.i.i134 to ptr
  %conv.i.i.i.i135 = zext i16 %34 to i32
  %clockBufferHead.i.i.i.i136 = getelementptr inbounds nuw i8, ptr %47, i64 20
  %bf.load.i.i.i.i137 = load i32, ptr %clockBufferHead.i.i.i.i136, align 4
  %bf.lshr.i.i.i.i138 = lshr i32 %bf.load.i.i.i.i137, 1
  %cmp3.not.i.i.i.i139 = icmp samesign ult i32 %bf.lshr.i.i.i.i138, %conv.i.i.i.i135
  br i1 %cmp3.not.i.i.i.i139, label %if.then4.i.i.i.i191, label %do.end9.i.i.i.i140

if.then4.i.i.i.i191:                              ; preds = %if.end.i.i.i112
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i.i193, i32 noundef %agg.tmp.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i194 = load i32, ptr %clockBufferHead.i.i.i.i136, align 4
  %.pre42.i.i.i.i195 = lshr i32 %bf.load13.pre.i.i.i.i194, 1
  br label %do.end9.i.i.i.i140

do.end9.i.i.i.i140:                               ; preds = %if.then4.i.i.i.i191, %if.end.i.i.i112
  %bf.lshr14.pre-phi.i.i.i.i141 = phi i32 [ %bf.lshr.i.i.i.i138, %if.end.i.i.i112 ], [ %.pre42.i.i.i.i195, %if.then4.i.i.i.i191 ]
  %and.i.i.i.i.i142 = and i64 %add9.i.i.i.i134, -1073741824
  %48 = inttoptr i64 %and.i.i.i.i.i142 to ptr
  %sub.i.i.i.i143 = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i141, %conv.i.i.i.i135
  %clockBufferSize.i.i.i.i144 = getelementptr inbounds nuw i8, ptr %48, i64 26
  %49 = load i16, ptr %clockBufferSize.i.i.i.i144, align 2, !tbaa !15
  %conv16.i.i.i.i145 = zext i16 %49 to i32
  %cmp17.i.i.i.i146 = icmp slt i32 %sub.i.i.i.i143, %conv16.i.i.i.i145
  br i1 %cmp17.i.i.i.i146, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i151, label %if.then18.i.i.i.i147

if.then18.i.i.i.i147:                             ; preds = %do.end9.i.i.i.i140
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i38.i.i.i.i193, i32 noundef %agg.tmp.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i150 = load i16, ptr %clockBufferSize.i.i.i.i144, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i151

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i151: ; preds = %if.then18.i.i.i.i147, %do.end9.i.i.i.i140
  %50 = phi i16 [ %.pre.i.i.i.i150, %if.then18.i.i.i.i147 ], [ %49, %do.end9.i.i.i.i140 ]
  %51 = urem i16 %34, %50
  %rem.i.i.i.i152 = zext i16 %51 to i64
  %vectorClock.i.i.i.i.i153 = getelementptr inbounds nuw i8, ptr %47, i64 30
  %numThreads.i.i.i.i.i154 = getelementptr inbounds nuw i8, ptr %48, i64 24
  %52 = load i16, ptr %numThreads.i.i.i.i.i154, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i155 = zext i16 %52 to i64
  %add.ptr.i.i.i.i.i156 = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i153, i64 %idx.ext.i.i.i.i.i155
  %mul.i8.i.i.i157 = mul nuw nsw i64 %idx.ext.i.i.i.i.i155, %rem.i.i.i.i152
  %add.ptr.i.i.i.i158 = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i156, i64 %mul.i8.i.i.i157
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i159

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i159: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i151, %do.body.i107
  %retval.0.i.i22.i160 = phi ptr [ %add.ptr.i.i.i.i158, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i151 ], [ null, %do.body.i107 ]
  %tobool.not.not.i.i161 = icmp eq ptr %retval.0.i.i22.i160, null
  br i1 %tobool.not.not.i.i161, label %cleanup.i.i177, label %if.then1.i.i162

if.then1.i.i162:                                  ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i159
  %53 = load i16, ptr %6, align 8, !tbaa !11
  %conv.i.i.i166 = zext i16 %53 to i32
  %cmp.not11.i.i.i167 = icmp eq i16 %53, 0
  br i1 %cmp.not11.i.i.i167, label %cleanup.i.i177, label %for.body.i.i.i168

for.body.i.i.i168:                                ; preds = %if.then1.i.i162, %for.body.i.i.i168
  %i.012.i.i.i169 = phi i32 [ %inc.i.i.i174, %for.body.i.i.i168 ], [ 0, %if.then1.i.i162 ]
  %idxprom.i.i.i170 = zext nneg i32 %i.012.i.i.i169 to i64
  %arrayidx.i.i.i171 = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i164, i64 %idxprom.i.i.i170
  %54 = load i16, ptr %arrayidx.i.i.i171, align 2, !tbaa !22
  %arrayidx3.i.i.i172 = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i160, i64 %idxprom.i.i.i170
  %55 = load i16, ptr %arrayidx3.i.i.i172, align 2, !tbaa !22
  %cmp5.i.i.i173 = icmp uge i16 %54, %55
  %inc.i.i.i174 = add nuw nsw i32 %i.012.i.i.i169, 1
  %exitcond.not.i.i.i175 = icmp ne i32 %inc.i.i.i174, %conv.i.i.i166
  %or.cond.not.i.i.i176 = select i1 %cmp5.i.i.i173, i1 %exitcond.not.i.i.i175, i1 false
  br i1 %or.cond.not.i.i.i176, label %for.body.i.i.i168, label %cleanup.i.i177, !llvm.loop !31

cleanup.i.i177:                                   ; preds = %for.body.i.i.i168, %if.then1.i.i162, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i159
  %retval.0.i23.i178 = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i159 ], [ true, %if.then1.i.i162 ], [ %cmp5.i.i.i173, %for.body.i.i.i168 ]
  br i1 %tobool.not.not.i.i161, label %cleanup.cont.i.i184, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i179

cleanup.cont.i.i184:                              ; preds = %cleanup.i.i177
  %bf.load.i.i186 = load i16, ptr %scope.i93, align 2
  %bf.clear.i.i187 = and i16 %bf.load.i.i186, 4095
  %idxprom.i.i188 = zext nneg i16 %bf.clear.i.i187 to i64
  %arrayidx.i.i189 = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i164, i64 %idxprom.i.i188
  %56 = load i16, ptr %arrayidx.i.i189, align 2, !tbaa !22
  %57 = load i16, ptr %arrayidx13, align 4, !tbaa !27
  %cmp7.i.i190 = icmp uge i16 %56, %57
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i179

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i179: ; preds = %cleanup.cont.i.i184, %cleanup.i.i177
  %retval.1.i.i180 = phi i1 [ %retval.0.i23.i178, %cleanup.i.i177 ], [ %cmp7.i.i190, %cleanup.cont.i.i184 ]
  br i1 %retval.1.i.i180, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206, label %if.then9.i181

if.then9.i181:                                    ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i179
  tail call void @__assertfail(ptr noundef nonnull @.str7, ptr noundef nonnull %cond.i38.i.i.i.i193, i32 noundef %agg.tmp.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206

_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit206: ; preds = %for.body11, %land.rhs.i.i101, %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i105, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i179, %if.then9.i181
  %inc = add nuw nsw i32 %iRead.0304, 1
  %exitcond.not = icmp eq i32 %inc, 4
  br i1 %exitcond.not, label %for.cond.cleanup10, label %for.body11, !llvm.loop !58

if.then19:                                        ; preds = %for.cond.cleanup
  %agg.tmp20.sroa.0.0.copyload = load ptr, ptr %loc, align 8, !tbaa !55
  %agg.tmp20.sroa.2.0.loc.sroa_idx = getelementptr inbounds nuw i8, ptr %loc, i64 8
  %agg.tmp20.sroa.2.0.copyload = load i32, ptr %agg.tmp20.sroa.2.0.loc.sroa_idx, align 8, !tbaa !3
  %clockBufferDirty.i = getelementptr inbounds nuw i8, ptr %0, i64 20
  %bf.load.i208 = load i32, ptr %clockBufferDirty.i, align 4
  %bf.clear.i = and i32 %bf.load.i208, 1
  %tobool.not.i = icmp eq i32 %bf.clear.i, 0
  br i1 %tobool.not.i, label %if.end.i217, label %if.then.i

if.then.i:                                        ; preds = %if.then19
  %vectorClock.i = getelementptr inbounds nuw i8, ptr %0, i64 30
  %58 = ptrtoint ptr %0 to i64
  %and.i.i.i = and i64 %58, -1073741824
  %59 = inttoptr i64 %and.i.i.i to ptr
  %clockBufferSize.i.i = getelementptr inbounds nuw i8, ptr %59, i64 26
  %60 = load i16, ptr %clockBufferSize.i.i, align 2, !tbaa !15
  %cmp.not.i.i = icmp eq i16 %60, 0
  br i1 %cmp.not.i.i, label %if.then.i.i, label %do.end.i.i

if.then.i.i:                                      ; preds = %if.then.i
  %cmp.i.i.i = icmp eq ptr %agg.tmp20.sroa.0.0.copyload, null
  %cond.i.i.i = select i1 %cmp.i.i.i, ptr @.str, ptr %agg.tmp20.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str10, ptr noundef nonnull %cond.i.i.i, i32 noundef %agg.tmp20.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load.i.pre.i = load i32, ptr %clockBufferDirty.i, align 4
  br label %do.end.i.i

do.end.i.i:                                       ; preds = %if.then.i.i, %if.then.i
  %bf.load.i.i212 = phi i32 [ %bf.load.i.pre.i, %if.then.i.i ], [ %bf.load.i208, %if.then.i ]
  %bf.lshr.i.i = lshr i32 %bf.load.i.i212, 1
  %add.i.i = add nuw i32 %bf.lshr.i.i, 1
  %cmp3.i.i = icmp ult i32 %bf.load.i.i212, 131070
  br i1 %cmp3.i.i, label %do.end10.i.i, label %if.then4.i.i

if.then4.i.i:                                     ; preds = %do.end.i.i
  %cmp.i35.i.i = icmp eq ptr %agg.tmp20.sroa.0.0.copyload, null
  %cond.i36.i.i = select i1 %cmp.i35.i.i, ptr @.str, ptr %agg.tmp20.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str11, ptr noundef nonnull %cond.i36.i.i, i32 noundef %agg.tmp20.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.end10.i.i

do.end10.i.i:                                     ; preds = %if.then4.i.i, %do.end.i.i
  %numThreads.i.i.i213 = getelementptr inbounds nuw i8, ptr %59, i64 24
  %61 = load i16, ptr %numThreads.i.i.i213, align 8, !tbaa !11
  %idx.ext.i.i.i = zext i16 %61 to i64
  %add.ptr.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i, i64 %idx.ext.i.i.i
  %62 = load i16, ptr %clockBufferSize.i.i, align 2, !tbaa !15
  %conv13.i.i = zext i16 %62 to i32
  %rem.i.i = urem i32 %add.i.i, %conv13.i.i
  %conv14.i.i = zext i16 %61 to i32
  %mul.i.i = mul nuw i32 %rem.i.i, %conv14.i.i
  %idx.ext.i.i = zext i32 %mul.i.i to i64
  %add.ptr.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i, i64 %idx.ext.i.i
  %cmp1738.not.i.i = icmp eq i16 %61, 0
  br i1 %cmp1738.not.i.i, label %_ZN4gsan12_GLOBAL__N_125appendClockBufferSnapshotEPNS_11ThreadStateEPKtNS_8LocationE.exit.i, label %for.body.i.i

for.body.i.i:                                     ; preds = %do.end10.i.i, %for.body.i.i
  %i.039.i.i = phi i32 [ %inc.i.i, %for.body.i.i ], [ 0, %do.end10.i.i ]
  %idxprom.i.i214 = zext nneg i32 %i.039.i.i to i64
  %arrayidx.i.i215 = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i, i64 %idxprom.i.i214
  %63 = load i16, ptr %arrayidx.i.i215, align 2, !tbaa !22
  %arrayidx19.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i, i64 %idxprom.i.i214
  store i16 %63, ptr %arrayidx19.i.i, align 2, !tbaa !22
  %inc.i.i = add nuw nsw i32 %i.039.i.i, 1
  %64 = load i16, ptr %numThreads.i.i.i213, align 8, !tbaa !11
  %conv16.i.i = zext i16 %64 to i32
  %cmp17.i.i = icmp samesign ult i32 %inc.i.i, %conv16.i.i
  br i1 %cmp17.i.i, label %for.body.i.i, label %_ZN4gsan12_GLOBAL__N_125appendClockBufferSnapshotEPNS_11ThreadStateEPKtNS_8LocationE.exit.i, !llvm.loop !59

_ZN4gsan12_GLOBAL__N_125appendClockBufferSnapshotEPNS_11ThreadStateEPKtNS_8LocationE.exit.i: ; preds = %for.body.i.i, %do.end10.i.i
  %bf.value.i.i = shl i32 %add.i.i, 1
  store i32 %bf.value.i.i, ptr %clockBufferDirty.i, align 4
  br label %_ZN4gsan12_GLOBAL__N_125publishCurrentVectorClockEPNS_11ThreadStateENS_8LocationE.exit

if.end.i217:                                      ; preds = %if.then19
  %bf.lshr.i218 = lshr exact i32 %bf.load.i208, 1
  br label %_ZN4gsan12_GLOBAL__N_125publishCurrentVectorClockEPNS_11ThreadStateENS_8LocationE.exit

_ZN4gsan12_GLOBAL__N_125publishCurrentVectorClockEPNS_11ThreadStateENS_8LocationE.exit: ; preds = %_ZN4gsan12_GLOBAL__N_125appendClockBufferSnapshotEPNS_11ThreadStateEPKtNS_8LocationE.exit.i, %if.end.i217
  %retval.0.in.i = phi i32 [ %add.i.i, %_ZN4gsan12_GLOBAL__N_125appendClockBufferSnapshotEPNS_11ThreadStateEPKtNS_8LocationE.exit.i ], [ %bf.lshr.i218, %if.end.i217 ]
  %retval.0.i216 = trunc i32 %retval.0.in.i to i16
  %65 = getelementptr i8, ptr %0, i64 28
  %.val85 = load i16, ptr %65, align 4, !tbaa !22
  %bf.value.i = and i16 %.val85, 4095
  %66 = trunc i24 %switch.downshift to i16
  %67 = shl i16 %66, 12
  %bf.set6.i = or disjoint i16 %bf.value.i, %67
  %bf.set9.i = or disjoint i16 %bf.set6.i, 16384
  br label %if.end36

if.else:                                          ; preds = %for.cond.cleanup
  %cells23 = getelementptr inbounds nuw i8, ptr %event, i64 8
  %68 = load ptr, ptr %cells23, align 8, !tbaa !49
  %writeClock25 = getelementptr inbounds nuw i8, ptr %68, i64 16
  %69 = load i32, ptr %writeClock25, align 4
  %previousWrite.sroa.0.0.extract.trunc = trunc i32 %69 to i16
  %previousWrite.sroa.4.0.extract.shift = lshr i32 %69, 16
  %previousWrite.sroa.4.0.extract.trunc = trunc nuw i32 %previousWrite.sroa.4.0.extract.shift to i16
  %70 = and i16 %previousWrite.sroa.4.0.extract.trunc, 16384
  %bf.cast.not = icmp eq i16 %70, 0
  br i1 %bf.cast.not, label %if.else32, label %if.then26

if.then26:                                        ; preds = %if.else
  %agg.tmp28.sroa.0.0.copyload = load ptr, ptr %loc, align 8, !tbaa !55
  %agg.tmp28.sroa.2.0.loc.sroa_idx = getelementptr inbounds nuw i8, ptr %loc, i64 8
  %agg.tmp28.sroa.2.0.copyload = load i32, ptr %agg.tmp28.sroa.2.0.loc.sroa_idx, align 8, !tbaa !3
  %71 = ptrtoint ptr %0 to i64
  %and.i.i.i224 = and i64 %71, -1073741824
  %72 = inttoptr i64 %and.i.i.i224 to ptr
  %bf.clear2.i.i = and i16 %previousWrite.sroa.4.0.extract.trunc, 4095
  %numSms.i.i.i225 = getelementptr inbounds nuw i8, ptr %72, i64 20
  %73 = load i16, ptr %numSms.i.i.i225, align 4, !tbaa !21
  %bf.clear2.i.i.frozen = freeze i16 %bf.clear2.i.i
  %.frozen317 = freeze i16 %73
  %div16.i.i.i = udiv i16 %bf.clear2.i.i.frozen, %.frozen317
  %74 = mul i16 %div16.i.i.i, %.frozen317
  %rem17.i.i.i.decomposed = sub i16 %bf.clear2.i.i.frozen, %74
  %globalsBase.i.i.i = getelementptr inbounds nuw i8, ptr %72, i64 8
  %75 = load i64, ptr %globalsBase.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i = zext nneg i16 %div16.i.i.i to i64
  %mul.i.i.i = shl nuw nsw i64 %conv5.i.i.i, 30
  %add.i.i.i = or disjoint i64 %mul.i.i.i, 39
  %ptr.biased.i.i.i.i.i = add i64 %add.i.i.i, %75
  %cond.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i, -8
  %76 = getelementptr i8, ptr %72, i64 24
  %globals.val.i.i.i = load i16, ptr %76, align 8, !tbaa !11
  %77 = getelementptr i8, ptr %72, i64 26
  %globals.val15.i.i.i = load i16, ptr %77, align 2, !tbaa !15
  %conv.i.i.i.i226 = zext i16 %globals.val15.i.i.i to i64
  %add.i.i.i.i227 = add nuw nsw i64 %conv.i.i.i.i226, 1
  %conv1.i.i.i.i = zext i16 %globals.val.i.i.i to i64
  %mul.i.i.i.i228 = shl nuw nsw i64 %conv1.i.i.i.i, 1
  %mul3.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i228, %add.i.i.i.i227
  %add4.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i, 32
  %conv7.i.i.i = zext nneg i16 %rem17.i.i.i.decomposed to i64
  %mul8.i.i.i = mul nuw nsw i64 %add4.i.i.i.i, %conv7.i.i.i
  %add9.i.i.i = add i64 %mul8.i.i.i, %cond.i.i.i.i.i
  %78 = inttoptr i64 %add9.i.i.i to ptr
  %conv.i.i.i229 = and i32 %69, 65535
  %cmp.not.i.i.i = icmp eq i16 %previousWrite.sroa.0.0.extract.trunc, 0
  br i1 %cmp.not.i.i.i, label %if.then.i.i.i, label %do.body1.i.i.i

if.then.i.i.i:                                    ; preds = %if.then26
  %cmp.i.i.i.i = icmp eq ptr %agg.tmp28.sroa.0.0.copyload, null
  %cond.i.i.i.i = select i1 %cmp.i.i.i.i, ptr @.str, ptr %agg.tmp28.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str3, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %agg.tmp28.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.body1.i.i.i

do.body1.i.i.i:                                   ; preds = %if.then.i.i.i, %if.then26
  %clockBufferHead.i.i.i = getelementptr inbounds nuw i8, ptr %78, i64 20
  %bf.load.i.i.i = load i32, ptr %clockBufferHead.i.i.i, align 4
  %bf.lshr.i.i.i = lshr i32 %bf.load.i.i.i, 1
  %cmp3.not.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i, %conv.i.i.i229
  br i1 %cmp3.not.i.i.i, label %if.then4.i.i.i, label %do.end9.i.i.i

if.then4.i.i.i:                                   ; preds = %do.body1.i.i.i
  %cmp.i37.i.i.i = icmp eq ptr %agg.tmp28.sroa.0.0.copyload, null
  %cond.i38.i.i.i = select i1 %cmp.i37.i.i.i, ptr @.str, ptr %agg.tmp28.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i, i32 noundef %agg.tmp28.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i = load i32, ptr %clockBufferHead.i.i.i, align 4
  %.pre42.i.i.i = lshr i32 %bf.load13.pre.i.i.i, 1
  br label %do.end9.i.i.i

do.end9.i.i.i:                                    ; preds = %if.then4.i.i.i, %do.body1.i.i.i
  %bf.lshr14.pre-phi.i.i.i = phi i32 [ %bf.lshr.i.i.i, %do.body1.i.i.i ], [ %.pre42.i.i.i, %if.then4.i.i.i ]
  %and.i.i.i.i230 = and i64 %add9.i.i.i, -1073741824
  %79 = inttoptr i64 %and.i.i.i.i230 to ptr
  %sub.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i, %conv.i.i.i229
  %clockBufferSize.i.i.i = getelementptr inbounds nuw i8, ptr %79, i64 26
  %80 = load i16, ptr %clockBufferSize.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i = zext i16 %80 to i32
  %cmp17.i.i.i = icmp slt i32 %sub.i.i.i, %conv16.i.i.i
  br i1 %cmp17.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i, label %if.then18.i.i.i

if.then18.i.i.i:                                  ; preds = %do.end9.i.i.i
  %cmp.i39.i.i.i = icmp eq ptr %agg.tmp28.sroa.0.0.copyload, null
  %cond.i40.i.i.i = select i1 %cmp.i39.i.i.i, ptr @.str, ptr %agg.tmp28.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i40.i.i.i, i32 noundef %agg.tmp28.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i = load i16, ptr %clockBufferSize.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i: ; preds = %if.then18.i.i.i, %do.end9.i.i.i
  %81 = phi i16 [ %.pre.i.i.i, %if.then18.i.i.i ], [ %80, %do.end9.i.i.i ]
  %82 = urem i16 %previousWrite.sroa.0.0.extract.trunc, %81
  %rem.i.i.i = zext i16 %82 to i64
  %vectorClock.i.i.i.i = getelementptr inbounds nuw i8, ptr %78, i64 30
  %numThreads.i.i.i.i = getelementptr inbounds nuw i8, ptr %79, i64 24
  %83 = load i16, ptr %numThreads.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i = zext i16 %83 to i64
  %add.ptr.i.i.i.i231 = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idx.ext.i.i.i.i
  %mul.i8.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i, %rem.i.i.i
  %add.ptr.i.i.i232 = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i231, i64 %mul.i8.i.i
  %84 = load i16, ptr %77, align 2, !tbaa !15
  %cmp.not.i.i235 = icmp eq i16 %84, 0
  br i1 %cmp.not.i.i235, label %if.then.i.i265, label %do.end.i.i236

if.then.i.i265:                                   ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i
  %cmp.i.i.i266 = icmp eq ptr %agg.tmp28.sroa.0.0.copyload, null
  %cond.i.i.i267 = select i1 %cmp.i.i.i266, ptr @.str, ptr %agg.tmp28.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str10, ptr noundef nonnull %cond.i.i.i267, i32 noundef %agg.tmp28.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.end.i.i236

do.end.i.i236:                                    ; preds = %if.then.i.i265, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i
  %clockBufferHead.i.i = getelementptr inbounds nuw i8, ptr %0, i64 20
  %bf.load.i.i237 = load i32, ptr %clockBufferHead.i.i, align 4
  %bf.lshr.i.i238 = lshr i32 %bf.load.i.i237, 1
  %add.i.i239 = add nuw i32 %bf.lshr.i.i238, 1
  %cmp3.i.i240 = icmp ult i32 %bf.load.i.i237, 131070
  br i1 %cmp3.i.i240, label %do.end10.i.i244, label %if.then4.i.i241

if.then4.i.i241:                                  ; preds = %do.end.i.i236
  %cmp.i35.i.i242 = icmp eq ptr %agg.tmp28.sroa.0.0.copyload, null
  %cond.i36.i.i243 = select i1 %cmp.i35.i.i242, ptr @.str, ptr %agg.tmp28.sroa.0.0.copyload
  tail call void @__assertfail(ptr noundef nonnull @.str11, ptr noundef nonnull %cond.i36.i.i243, i32 noundef %agg.tmp28.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.end10.i.i244

do.end10.i.i244:                                  ; preds = %if.then4.i.i241, %do.end.i.i236
  %vectorClock.i.i.i245 = getelementptr inbounds nuw i8, ptr %0, i64 30
  %85 = load i16, ptr %76, align 8, !tbaa !11
  %idx.ext.i.i.i247 = zext i16 %85 to i64
  %add.ptr.i.i12.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i245, i64 %idx.ext.i.i.i247
  %86 = load i16, ptr %77, align 2, !tbaa !15
  %conv13.i.i248 = zext i16 %86 to i32
  %rem.i.i249 = urem i32 %add.i.i239, %conv13.i.i248
  %conv14.i.i250 = zext i16 %85 to i32
  %mul.i.i251 = mul nuw i32 %rem.i.i249, %conv14.i.i250
  %idx.ext.i.i252 = zext i32 %mul.i.i251 to i64
  %add.ptr.i.i253 = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i12.i, i64 %idx.ext.i.i252
  %cmp1738.not.i.i254 = icmp eq i16 %85, 0
  br i1 %cmp1738.not.i.i254, label %_ZN4gsan12_GLOBAL__N_128propagateClockBufferSnapshotEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit, label %for.body.i.i255

for.body.i.i255:                                  ; preds = %do.end10.i.i244, %for.body.i.i255
  %i.039.i.i256 = phi i32 [ %inc.i.i260, %for.body.i.i255 ], [ 0, %do.end10.i.i244 ]
  %idxprom.i.i257 = zext nneg i32 %i.039.i.i256 to i64
  %arrayidx.i.i258 = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i232, i64 %idxprom.i.i257
  %87 = load i16, ptr %arrayidx.i.i258, align 2, !tbaa !22
  %arrayidx19.i.i259 = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i253, i64 %idxprom.i.i257
  store i16 %87, ptr %arrayidx19.i.i259, align 2, !tbaa !22
  %inc.i.i260 = add nuw nsw i32 %i.039.i.i256, 1
  %88 = load i16, ptr %76, align 8, !tbaa !11
  %conv16.i.i261 = zext i16 %88 to i32
  %cmp17.i.i262 = icmp samesign ult i32 %inc.i.i260, %conv16.i.i261
  br i1 %cmp17.i.i262, label %for.body.i.i255, label %_ZN4gsan12_GLOBAL__N_128propagateClockBufferSnapshotEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit, !llvm.loop !59

_ZN4gsan12_GLOBAL__N_128propagateClockBufferSnapshotEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit: ; preds = %for.body.i.i255, %do.end10.i.i244
  %bf.value.i.i264 = shl i32 %add.i.i239, 1
  %conv22.i.i = trunc i32 %add.i.i239 to i16
  %bf.set.i = or disjoint i32 %bf.value.i.i264, 1
  store i32 %bf.set.i, ptr %clockBufferHead.i.i, align 4
  %89 = getelementptr i8, ptr %0, i64 28
  %.val = load i16, ptr %89, align 4, !tbaa !22
  %bf.value.i270 = and i16 %.val, 4095
  %90 = trunc i24 %switch.downshift to i16
  %91 = shl i16 %90, 12
  %bf.set6.i272 = or disjoint i16 %bf.value.i270, %91
  %bf.set9.i273 = or disjoint i16 %bf.set6.i272, 16384
  br label %if.end36

if.else32:                                        ; preds = %if.else
  %threadId.i276 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %92 = load i16, ptr %threadId.i276, align 4, !tbaa !22
  %vectorClock.i277 = getelementptr inbounds nuw i8, ptr %0, i64 30
  %idxprom.i = zext i16 %92 to i64
  %arrayidx.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i277, i64 %idxprom.i
  %93 = load i16, ptr %arrayidx.i, align 2, !tbaa !22
  %bf.value.i278 = and i16 %92, 4095
  %94 = trunc i24 %switch.downshift to i16
  %95 = shl i16 %94, 12
  %bf.set6.i280 = or disjoint i16 %bf.value.i278, %95
  br label %if.end36

if.end36:                                         ; preds = %_ZN4gsan12_GLOBAL__N_128propagateClockBufferSnapshotEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit, %if.else32, %_ZN4gsan12_GLOBAL__N_125publishCurrentVectorClockEPNS_11ThreadStateENS_8LocationE.exit
  %retval.0.i216.pn = phi i16 [ %retval.0.i216, %_ZN4gsan12_GLOBAL__N_125publishCurrentVectorClockEPNS_11ThreadStateENS_8LocationE.exit ], [ %conv22.i.i, %_ZN4gsan12_GLOBAL__N_128propagateClockBufferSnapshotEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit ], [ %93, %if.else32 ]
  %bf.set9.i.pn = phi i16 [ %bf.set9.i, %_ZN4gsan12_GLOBAL__N_125publishCurrentVectorClockEPNS_11ThreadStateENS_8LocationE.exit ], [ %bf.set9.i273, %_ZN4gsan12_GLOBAL__N_128propagateClockBufferSnapshotEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit ], [ %bf.set6.i280, %if.else32 ]
  %96 = load i8, ptr %numCells, align 8, !tbaa !48
  %cmp42307.not = icmp eq i8 %96, 0
  br i1 %cmp42307.not, label %for.cond.cleanup43, label %for.body44.lr.ph

for.body44.lr.ph:                                 ; preds = %if.end36
  %cells45 = getelementptr inbounds nuw i8, ptr %event, i64 8
  br label %for.body44

for.cond.cleanup43:                               ; preds = %for.body44, %if.end36
  br i1 %.not, label %if.end56, label %if.then53

for.body44:                                       ; preds = %for.body44.lr.ph, %for.body44
  %i37.0308 = phi i8 [ 0, %for.body44.lr.ph ], [ %inc50, %for.body44 ]
  %idxprom46 = zext i8 %i37.0308 to i64
  %arrayidx47 = getelementptr inbounds nuw [8 x i8], ptr %cells45, i64 %idxprom46
  %97 = load ptr, ptr %arrayidx47, align 8, !tbaa !49
  %writeClock48 = getelementptr inbounds nuw i8, ptr %97, i64 16
  store i16 %retval.0.i216.pn, ptr %writeClock48, align 4, !tbaa !22
  %newWriteClock.sroa.6.0.writeClock48.sroa_idx = getelementptr inbounds nuw i8, ptr %97, i64 18
  store i16 %bf.set9.i.pn, ptr %newWriteClock.sroa.6.0.writeClock48.sroa_idx, align 2, !tbaa !23
  %inc50 = add nuw i8 %i37.0308, 1
  %98 = load i8, ptr %numCells, align 8, !tbaa !48
  %cmp42 = icmp ult i8 %inc50, %98
  br i1 %cmp42, label %for.body44, label %for.cond.cleanup43, !llvm.loop !60

if.then53:                                        ; preds = %for.cond.cleanup43
  %threadId.i284 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %99 = load i16, ptr %threadId.i284, align 4, !tbaa !22
  %vectorClock.i285 = getelementptr inbounds nuw i8, ptr %0, i64 30
  %idxprom.i286 = zext i16 %99 to i64
  %arrayidx.i287 = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i285, i64 %idxprom.i286
  %100 = load i16, ptr %arrayidx.i287, align 2, !tbaa !22
  %cmp.not.i288 = icmp eq i16 %100, -1
  br i1 %cmp.not.i288, label %if.then.i293, label %_ZN4gsan12_GLOBAL__N_120incrementThreadEpochEPNS_11ThreadStateENS_8LocationE.exit

if.then.i293:                                     ; preds = %if.then53
  %agg.tmp54.sroa.0.0.copyload = load ptr, ptr %loc, align 8, !tbaa !55
  %cmp.i.i296 = icmp eq ptr %agg.tmp54.sroa.0.0.copyload, null
  %cond.i.i297 = select i1 %cmp.i.i296, ptr @.str, ptr %agg.tmp54.sroa.0.0.copyload
  %agg.tmp54.sroa.2.0.loc.sroa_idx = getelementptr inbounds nuw i8, ptr %loc, i64 8
  %agg.tmp54.sroa.2.0.copyload = load i32, ptr %agg.tmp54.sroa.2.0.loc.sroa_idx, align 8, !tbaa !3
  tail call void @__assertfail(ptr noundef nonnull @.str6, ptr noundef nonnull %cond.i.i297, i32 noundef %agg.tmp54.sroa.2.0.copyload, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i = load i16, ptr %arrayidx.i287, align 2, !tbaa !22
  br label %_ZN4gsan12_GLOBAL__N_120incrementThreadEpochEPNS_11ThreadStateENS_8LocationE.exit

_ZN4gsan12_GLOBAL__N_120incrementThreadEpochEPNS_11ThreadStateENS_8LocationE.exit: ; preds = %if.then53, %if.then.i293
  %101 = phi i16 [ %.pre.i, %if.then.i293 ], [ %100, %if.then53 ]
  %add.i = add i16 %101, 1
  store i16 %add.i, ptr %arrayidx.i287, align 2, !tbaa !22
  %clockBufferDirty.i290 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %bf.load.i291 = load i32, ptr %clockBufferDirty.i290, align 4
  %bf.set.i292 = or i32 %bf.load.i291, 1
  store i32 %bf.set.i292, ptr %clockBufferDirty.i290, align 4
  br label %if.end56

if.end56:                                         ; preds = %for.cond.cleanup43, %_ZN4gsan12_GLOBAL__N_120incrementThreadEpochEPNS_11ThreadStateENS_8LocationE.exit, %switch.lookup
  %102 = load ptr, ptr %event, align 8, !tbaa !45
  %cmp.i298 = icmp eq ptr %102, null
  br i1 %cmp.i298, label %return, label %for.cond.preheader.i

for.cond.preheader.i:                             ; preds = %if.end56
  %numCells.i = getelementptr inbounds nuw i8, ptr %event, i64 32
  %103 = load i8, ptr %numCells.i, align 8, !tbaa !48
  %cmp210.not.i = icmp eq i8 %103, 0
  br i1 %cmp210.not.i, label %for.cond.cleanup.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %for.cond.preheader.i
  %cells.i = getelementptr inbounds nuw i8, ptr %event, i64 8
  br label %for.body.i

for.cond.cleanup.loopexit.i:                      ; preds = %for.body.i
  %.pre.i301 = load ptr, ptr %event, align 8, !tbaa !45
  br label %for.cond.cleanup.i

for.cond.cleanup.i:                               ; preds = %for.cond.cleanup.loopexit.i, %for.cond.preheader.i
  %104 = phi ptr [ %.pre.i301, %for.cond.cleanup.loopexit.i ], [ %102, %for.cond.preheader.i ]
  %lock.i = getelementptr inbounds nuw i8, ptr %104, i64 24
  %105 = atomicrmw and ptr %lock.i, i32 2147483647 syncscope("block") release, align 4
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(33) %event, i8 0, i64 33, i1 false)
  br label %return

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  %i.011.i = phi i8 [ 0, %for.body.lr.ph.i ], [ %inc.i, %for.body.i ]
  %idxprom.i299 = zext i8 %i.011.i to i64
  %arrayidx.i300 = getelementptr inbounds nuw [8 x i8], ptr %cells.i, i64 %idxprom.i299
  %106 = load ptr, ptr %arrayidx.i300, align 8, !tbaa !49
  %lock.i.i = getelementptr inbounds nuw i8, ptr %106, i64 22
  store atomic i16 0, ptr %lock.i.i release, align 2
  %inc.i = add nuw i8 %i.011.i, 1
  %107 = load i8, ptr %numCells.i, align 8, !tbaa !48
  %cmp2.i = icmp ult i8 %inc.i, %107
  br i1 %cmp2.i, label %for.body.i, label %for.cond.cleanup.loopexit.i, !llvm.loop !61

return:                                           ; preds = %for.cond.cleanup.i, %if.end56, %entry, %lor.lhs.false
  ret void
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_atomic_begin_scalar(ptr noundef %globalState, ptr noundef captures(none) initializes((0, 33)) %eventState, i32 noundef %pred, i64 noundef %address, i32 noundef %bytesPerElem, i32 noundef %sem, i32 noundef %scope, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %cmp.not = icmp eq i32 %pred, 0
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(33) %eventState, i8 0, i64 33, i1 false)
  br i1 %cmp.not, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, label %if.end.i

if.end.i:                                         ; preds = %entry
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i.i.i = add i64 %1, 39
  %cond.i.i.i.i = and i64 %ptr.biased.i.i.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val22.i.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i.i = zext i16 %globals.val22.i.i to i64
  %add.i.i.i = add nuw nsw i64 %conv.i.i.i, 1
  %conv1.i.i.i = zext i16 %globals.val.i.i to i64
  %mul.i.i.i = shl nuw nsw i64 %conv1.i.i.i, 1
  %mul3.i.i.i = mul nuw nsw i64 %mul.i.i.i, %add.i.i.i
  %add4.i.i.i = add nuw nsw i64 %mul3.i.i.i, 32
  %conv.i.i = zext i32 %0 to i64
  %mul.i.i = mul i64 %add4.i.i.i, %conv.i.i
  %add.i.i = add i64 %mul.i.i, %cond.i.i.i.i
  %4 = inttoptr i64 %add.i.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i.i = icmp eq ptr %5, null
  br i1 %cmp.i.i, label %if.then.i.i, label %if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i

if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i: ; preds = %if.end.i
  %reserveBase1.i.phi.trans.insert.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  %.pre.i = load i64, ptr %reserveBase1.i.phi.trans.insert.i, align 8, !tbaa !19
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

if.then.i.i:                                      ; preds = %if.end.i
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase4.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase4.i.i, align 8, !tbaa !19
  %numReads.i.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i.i, align 8, !tbaa !3
  %clockBufferDirty.i.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i.i, align 4
  %globalsBase1.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i.i, align 8, !tbaa !20
  %sub.i.i.i = sub i64 %1, %7
  %div6.i.i.i = lshr i64 %sub.i.i.i, 30
  %numSms.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i.i, align 4, !tbaa !21
  %conv.i23.i.i = zext i16 %8 to i64
  %mul.i24.i.i = mul nuw nsw i64 %div6.i.i.i, %conv.i23.i.i
  %add.i25.i.i = add nuw nsw i64 %mul.i24.i.i, %conv.i.i
  %conv3.i.i.i = trunc i64 %add.i25.i.i to i16
  %threadId.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i.i, ptr %threadId.i.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i: ; preds = %if.then.i.i, %if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i
  %9 = phi i64 [ %.pre.i, %if.end._ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit_crit_edge.i ], [ %6, %if.then.i.i ]
  %conv.i46.i = sext i32 %bytesPerElem to i64
  %add.i47.i = add i64 %address, %conv.i46.i
  %sub.i.i48.i = and i64 %address, -4
  %rem3.i.i.i = and i64 %add.i47.i, 3
  %cmp.i.i.i = icmp eq i64 %rem3.i.i.i, 0
  %sub5.i.i.i = sub nuw nsw i64 4, %rem3.i.i.i
  %cond.i.i.i = select i1 %cmp.i.i.i, i64 0, i64 %sub5.i.i.i
  %add.i.i49.i = add i64 %cond.i.i.i, %add.i47.i
  %cmp62.i.i = icmp ult i64 %sub.i.i48.i, %add.i.i49.i
  br i1 %cmp62.i.i, label %for.body.i.i.preheader, label %for.cond.cleanup.i.i

for.body.i.i.preheader:                           ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i
  %10 = add i64 %sub.i.i48.i, 4
  %umax = tail call i64 @llvm.umax.i64(i64 %add.i.i49.i, i64 %10)
  %11 = xor i64 %sub.i.i48.i, -1
  %12 = add i64 %umax, %11
  %13 = lshr i64 %12, 2
  %14 = add nuw nsw i64 %13, 1
  %min.iters.check = icmp eq i64 %13, 0
  br i1 %min.iters.check, label %for.body.i.i.preheader18, label %vector.ph

vector.ph:                                        ; preds = %for.body.i.i.preheader
  %n.vec = and i64 %14, 9223372036854775806
  %15 = shl i64 %n.vec, 2
  %16 = add i64 %sub.i.i48.i, %15
  %broadcast.splatinsert = insertelement <2 x i64> poison, i64 %9, i64 0
  %broadcast.splat = shufflevector <2 x i64> %broadcast.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert15 = insertelement <2 x i64> poison, i64 %sub.i.i48.i, i64 0
  %broadcast.splat16 = shufflevector <2 x i64> %broadcast.splatinsert15, <2 x i64> poison, <2 x i32> zeroinitializer
  %induction = add <2 x i64> %broadcast.splat16, <i64 0, i64 4>
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <2 x i64> [ %induction, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %vec.phi = phi <2 x i8> [ zeroinitializer, %vector.ph ], [ %20, %vector.body ]
  %17 = and <2 x i64> %vec.ind, splat (i64 -1099511627776)
  %18 = icmp eq <2 x i64> %17, %broadcast.splat
  %19 = zext <2 x i1> %18 to <2 x i8>
  %20 = add <2 x i8> %vec.phi, %19
  %index.next = add nuw i64 %index, 2
  %vec.ind.next = add <2 x i64> %vec.ind, splat (i64 8)
  %21 = icmp eq i64 %index.next, %n.vec
  br i1 %21, label %middle.block, label %vector.body, !llvm.loop !62

middle.block:                                     ; preds = %vector.body
  %22 = tail call i8 @llvm.vector.reduce.add.v2i8(<2 x i8> %20)
  %cmp.n = icmp eq i64 %14, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup.i.i, label %for.body.i.i.preheader18

for.body.i.i.preheader18:                         ; preds = %for.body.i.i.preheader, %middle.block
  %addr.064.i.i.ph = phi i64 [ %sub.i.i48.i, %for.body.i.i.preheader ], [ %16, %middle.block ]
  %numCells.063.i.i.ph = phi i8 [ 0, %for.body.i.i.preheader ], [ %22, %middle.block ]
  br label %for.body.i.i

for.cond.cleanup.i.i:                             ; preds = %for.body.i.i, %middle.block, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i
  %numCells.0.lcssa.i.i = phi i8 [ 0, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i ], [ %22, %middle.block ], [ %spec.select.i.i, %for.body.i.i ]
  %cmp7.i.i = icmp ult i8 %numCells.0.lcssa.i.i, 4
  br i1 %cmp7.i.i, label %do.end.i.i, label %if.then8.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.preheader18, %for.body.i.i
  %addr.064.i.i = phi i64 [ %add5.i.i, %for.body.i.i ], [ %addr.064.i.i.ph, %for.body.i.i.preheader18 ]
  %numCells.063.i.i = phi i8 [ %spec.select.i.i, %for.body.i.i ], [ %numCells.063.i.i.ph, %for.body.i.i.preheader18 ]
  %and.i.i.i.i = and i64 %addr.064.i.i, -1099511627776
  %cmp.i53.i.i = icmp eq i64 %and.i.i.i.i, %9
  %inc.i.i = zext i1 %cmp.i53.i.i to i8
  %spec.select.i.i = add i8 %numCells.063.i.i, %inc.i.i
  %add5.i.i = add i64 %addr.064.i.i, 4
  %cmp.i51.i = icmp ult i64 %add5.i.i, %add.i.i49.i
  br i1 %cmp.i51.i, label %for.body.i.i, label %for.cond.cleanup.i.i, !llvm.loop !63

if.then8.i.i:                                     ; preds = %for.cond.cleanup.i.i
  %cmp.i54.i.i = icmp eq ptr %file, null
  %cond.i55.i.i = select i1 %cmp.i54.i.i, ptr @.str, ptr %file
  tail call void @__assertfail(ptr noundef nonnull @.str9, ptr noundef nonnull %cond.i55.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.end.i.i

do.end.i.i:                                       ; preds = %if.then8.i.i, %for.cond.cleanup.i.i
  %cmp13.i.i = icmp eq i8 %numCells.0.lcssa.i.i, 0
  br i1 %cmp13.i.i, label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i, label %if.end15.i.i

if.end15.i.i:                                     ; preds = %do.end.i.i
  %lock.i.i = getelementptr inbounds nuw i8, ptr %4, i64 24
  br label %while.cond.i.i.i

while.cond.i.i.i:                                 ; preds = %while.cond.i.i.i, %if.end15.i.i
  %23 = cmpxchg weak ptr %lock.i.i, i32 0, i32 -2147483648 syncscope("block") acquire monotonic, align 4
  %24 = extractvalue { i32, i1 } %23, 1
  br i1 %24, label %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i, label %while.cond.i.i.i, !llvm.loop !44

_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i: ; preds = %while.cond.i.i.i
  store ptr %4, ptr %eventState, align 8, !tbaa !45
  %numCells16.i.i = getelementptr inbounds nuw i8, ptr %eventState, i64 32
  store i8 0, ptr %numCells16.i.i, align 8, !tbaa !48
  br i1 %cmp62.i.i, label %for.body23.lr.ph.i.i, label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i

for.body23.lr.ph.i.i:                             ; preds = %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i
  %cells.i.i = getelementptr inbounds nuw i8, ptr %eventState, i64 8
  %invariant.op = sub i64 -549755813888, %9
  br label %for.body23.i.i

for.body23.i.i:                                   ; preds = %for.inc31.i.i, %for.body23.lr.ph.i.i
  %addr17.066.i.i = phi i64 [ %sub.i.i48.i, %for.body23.lr.ph.i.i ], [ %add32.i.i, %for.inc31.i.i ]
  %and.i.i56.i.i = and i64 %addr17.066.i.i, -1099511627776
  %cmp.i57.i.i = icmp eq i64 %and.i.i56.i.i, %9
  br i1 %cmp.i57.i.i, label %if.end26.i.i, label %for.inc31.i.i

if.end26.i.i:                                     ; preds = %for.body23.i.i
  %sub.i59.reass.i.reass.i.reass.reass = add i64 %addr17.066.i.i, %invariant.op
  %div4.i.i.i = lshr exact i64 %sub.i59.reass.i.reass.i.reass.reass, 2
  %mul.i.i50.i = mul i64 %div4.i.i.i, 24
  %add.i60.i.i = add i64 %mul.i.i50.i, %9
  %25 = inttoptr i64 %add.i60.i.i to ptr
  %lock.i.i.i = getelementptr inbounds nuw i8, ptr %25, i64 22
  br label %while.cond.i61.i.i

while.cond.i61.i.i:                               ; preds = %while.cond.i61.i.i, %if.end26.i.i
  %26 = cmpxchg weak ptr %lock.i.i.i, i16 0, i16 1 acquire monotonic, align 2
  %27 = extractvalue { i16, i1 } %26, 1
  br i1 %27, label %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, label %while.cond.i61.i.i, !llvm.loop !26

_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i: ; preds = %while.cond.i61.i.i
  %28 = load i8, ptr %numCells16.i.i, align 8, !tbaa !48
  %inc30.i.i = add i8 %28, 1
  store i8 %inc30.i.i, ptr %numCells16.i.i, align 8, !tbaa !48
  %idxprom.i.i = zext i8 %28 to i64
  %arrayidx.i.i = getelementptr inbounds nuw [8 x i8], ptr %cells.i.i, i64 %idxprom.i.i
  store ptr %25, ptr %arrayidx.i.i, align 8, !tbaa !49
  br label %for.inc31.i.i

for.inc31.i.i:                                    ; preds = %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, %for.body23.i.i
  %add32.i.i = add i64 %addr17.066.i.i, 4
  %cmp21.i.i = icmp ult i64 %add32.i.i, %add.i.i49.i
  br i1 %cmp21.i.i, label %for.body23.i.i, label %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i, !llvm.loop !51

_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i: ; preds = %for.inc31.i.i, %_ZN4gsan12_GLOBAL__N_118rwLockAcquireWriteERj.exit.i.i, %do.end.i.i
  %29 = load ptr, ptr %eventState, align 8, !tbaa !45
  %cmp.i = icmp eq ptr %29, null
  br i1 %cmp.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, label %if.end2.i

if.end2.i:                                        ; preds = %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i
  %switch.tableidx.i.i = add i32 %sem, -1
  %30 = icmp ult i32 %switch.tableidx.i.i, 4
  br i1 %30, label %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i, label %sw.default.i.i

sw.default.i.i:                                   ; preds = %if.end2.i
  tail call void @llvm.trap()
  unreachable

_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i: ; preds = %if.end2.i
  %switch.idx.cast.i.i = trunc nuw nsw i32 %switch.tableidx.i.i to i8
  %switch.tableidx = add i32 %scope, -1
  %31 = icmp ult i32 %switch.tableidx, 3
  br i1 %31, label %switch.lookup, label %sw.default.i52.i

sw.default.i52.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i
  tail call void @llvm.trap()
  unreachable

switch.lookup:                                    ; preds = %_ZN4gsan12_GLOBAL__N_115decodeAtomicSemEj.exit.i
  %switch.cast = trunc nuw i32 %switch.tableidx to i24
  %switch.shiftamt = shl nuw nsw i24 %switch.cast, 3
  %switch.downshift = lshr i24 196866, %switch.shiftamt
  %switch.masked = trunc i24 %switch.downshift to i8
  %numCells.i = getelementptr inbounds nuw i8, ptr %eventState, i64 32
  %32 = load i8, ptr %numCells.i, align 8, !tbaa !48
  %cmp6113.not.i = icmp eq i8 %32, 0
  br i1 %cmp6113.not.i, label %for.cond.cleanup.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %switch.lookup
  %cells.i = getelementptr inbounds nuw i8, ptr %eventState, i64 8
  %threadId.i56.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  %and.i.i.i = and i64 %add.i.i, -1073741824
  %33 = inttoptr i64 %and.i.i.i to ptr
  %numSms.i.i.i.i = getelementptr inbounds nuw i8, ptr %33, i64 20
  %globalsBase.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %33, i64 8
  %34 = getelementptr i8, ptr %33, i64 24
  %35 = getelementptr i8, ptr %33, i64 26
  %cmp.i37.i.i.i.i.i = icmp eq ptr %file, null
  %cond.i38.i.i.i.i.i = select i1 %cmp.i37.i.i.i.i.i, ptr @.str, ptr %file
  %vectorClock.i.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %36 = trunc i24 %switch.downshift to i16
  %37 = shl i16 %36, 12
  %numReads18.i.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  %rngSeed.i.i = getelementptr inbounds nuw i8, ptr %33, i64 16
  br label %for.body.i

for.cond.cleanup.loopexit.i:                      ; preds = %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i
  %38 = icmp eq i8 %72, 0
  br label %for.cond.cleanup.i

for.cond.cleanup.i:                               ; preds = %for.cond.cleanup.loopexit.i, %switch.lookup
  %cmp15115.not.i = phi i1 [ %38, %for.cond.cleanup.loopexit.i ], [ true, %switch.lookup ]
  switch i8 %switch.idx.cast.i.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit [
    i8 3, label %for.cond11.preheader.i
    i8 1, label %for.cond11.preheader.i
  ]

for.cond11.preheader.i:                           ; preds = %for.cond.cleanup.i, %for.cond.cleanup.i
  br i1 %cmp15115.not.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, label %for.body17.lr.ph.i

for.body17.lr.ph.i:                               ; preds = %for.cond11.preheader.i
  %cells19.i = getelementptr inbounds nuw i8, ptr %eventState, i64 8
  %threadId.i71.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  %and.i.i72.i = and i64 %add.i.i, -1073741824
  %39 = inttoptr i64 %and.i.i72.i to ptr
  %numSms.i.i.i105.i = getelementptr inbounds nuw i8, ptr %39, i64 20
  %globalsBase.i.i.i.i = getelementptr inbounds nuw i8, ptr %39, i64 8
  %40 = getelementptr i8, ptr %39, i64 24
  %41 = getelementptr i8, ptr %39, i64 26
  %cmp.i.i.i.i.i = icmp eq ptr %file, null
  %cond.i.i.i.i.i = select i1 %cmp.i.i.i.i.i, ptr @.str, ptr %file
  %vectorClock.i.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %clockBufferDirty.i95.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  br label %for.body17.i

for.body.i:                                       ; preds = %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i, %for.body.lr.ph.i
  %i.0114.i = phi i8 [ 0, %for.body.lr.ph.i ], [ %inc.i, %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i ]
  %idxprom.i = zext i8 %i.0114.i to i64
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %cells.i, i64 %idxprom.i
  %42 = load ptr, ptr %arrayidx.i, align 8, !tbaa !49
  %writeClock.i = getelementptr inbounds nuw i8, ptr %42, i64 16
  %43 = load i32, ptr %writeClock.i, align 4
  %write.sroa.0.0.extract.trunc.i = trunc i32 %43 to i16
  %write.sroa.5.0.extract.shift.i = lshr i32 %43, 16
  %write.sroa.5.0.extract.trunc.i = trunc nuw i32 %write.sroa.5.0.extract.shift.i to i16
  %cmp.i55.i = icmp eq i16 %write.sroa.0.0.extract.trunc.i, 0
  br i1 %cmp.i55.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i, label %if.end.i.i

if.end.i.i:                                       ; preds = %for.body.i
  %44 = load i16, ptr %threadId.i56.i, align 4, !tbaa !22
  %bf.lshr.i.i = lshr i16 %write.sroa.5.0.extract.trunc.i, 12
  %45 = trunc nuw nsw i16 %bf.lshr.i.i to i8
  %bf.cast.i.i = and i8 %45, 3
  %bf.clear3.i.i = and i16 %write.sroa.5.0.extract.trunc.i, 4095
  %cmp.i9.i.not.i.i = icmp eq i8 %bf.cast.i.i, 0
  br i1 %cmp.i9.i.not.i.i, label %do.body.i.i, label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %if.end.i.i
  switch i8 %switch.masked, label %if.end.i.i.i.unreachabledefault [
    i8 1, label %sw.bb.i.i.i.i
    i8 2, label %sw.bb2.i.i.i.i
    i8 3, label %land.rhs.i.i.i
  ]

sw.bb.i.i.i.i:                                    ; preds = %if.end.i.i.i
  %cmp.i10.i.i.i = icmp eq i16 %44, %bf.clear3.i.i
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i

sw.bb2.i.i.i.i:                                   ; preds = %if.end.i.i.i
  %46 = load i16, ptr %numSms.i.i.i.i, align 4, !tbaa !21
  %47 = udiv i16 %44, %46
  %48 = udiv i16 %bf.clear3.i.i, %46
  %cmp9.i.i.i.i = icmp eq i16 %47, %48
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i

if.end.i.i.i.unreachabledefault:                  ; preds = %if.end.i.i.i
  unreachable

default.unreachable:                              ; preds = %land.rhs.i.i.i, %land.rhs.i.i75.i, %if.end.i.i74.i
  unreachable

_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i: ; preds = %sw.bb2.i.i.i.i, %sw.bb.i.i.i.i
  %retval.0.i.i.i.i = phi i1 [ %cmp9.i.i.i.i, %sw.bb2.i.i.i.i ], [ %cmp.i10.i.i.i, %sw.bb.i.i.i.i ]
  br i1 %retval.0.i.i.i.i, label %land.rhs.i.i.i, label %do.body.i.i

land.rhs.i.i.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i, %if.end.i.i.i
  switch i8 %bf.cast.i.i, label %default.unreachable [
    i8 1, label %sw.bb.i16.i.i.i
    i8 2, label %sw.bb2.i13.i.i.i
    i8 3, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i
  ]

sw.bb.i16.i.i.i:                                  ; preds = %land.rhs.i.i.i
  %cmp.i17.i.i.i = icmp eq i16 %44, %bf.clear3.i.i
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i

sw.bb2.i13.i.i.i:                                 ; preds = %land.rhs.i.i.i
  %49 = load i16, ptr %numSms.i.i.i.i, align 4, !tbaa !21
  %50 = udiv i16 %44, %49
  %51 = udiv i16 %bf.clear3.i.i, %49
  %cmp9.i15.i.i.i = icmp eq i16 %50, %51
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i

_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i: ; preds = %sw.bb2.i13.i.i.i, %sw.bb.i16.i.i.i
  %retval.0.i.i.i = phi i1 [ %cmp.i17.i.i.i, %sw.bb.i16.i.i.i ], [ %cmp9.i15.i.i.i, %sw.bb2.i13.i.i.i ]
  br i1 %retval.0.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i, label %do.body.i.i

do.body.i.i:                                      ; preds = %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i, %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i.i, %if.end.i.i
  %52 = and i16 %write.sroa.5.0.extract.trunc.i, 16384
  %bf.cast.not.i.i.i.i = icmp eq i16 %52, 0
  br i1 %bf.cast.not.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i, label %if.end.i.i.i.i

if.end.i.i.i.i:                                   ; preds = %do.body.i.i
  %53 = load i16, ptr %numSms.i.i.i.i, align 4, !tbaa !21
  %bf.clear3.i.i.frozen = freeze i16 %bf.clear3.i.i
  %.frozen = freeze i16 %53
  %div16.i.i.i.i.i = udiv i16 %bf.clear3.i.i.frozen, %.frozen
  %54 = mul i16 %div16.i.i.i.i.i, %.frozen
  %rem17.i.i.i.i.i.decomposed = sub i16 %bf.clear3.i.i.frozen, %54
  %55 = load i64, ptr %globalsBase.i.i.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i.i = zext nneg i16 %div16.i.i.i.i.i to i64
  %mul.i.i.i.i.i = shl nuw nsw i64 %conv5.i.i.i.i.i, 30
  %add.i.i.i.i.i = or disjoint i64 %mul.i.i.i.i.i, 39
  %ptr.biased.i.i.i.i.i.i.i = add i64 %add.i.i.i.i.i, %55
  %cond.i.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i.i = load i16, ptr %34, align 8, !tbaa !11
  %globals.val15.i.i.i.i.i = load i16, ptr %35, align 2, !tbaa !15
  %conv.i.i.i.i.i.i = zext i16 %globals.val15.i.i.i.i.i to i64
  %add.i.i.i.i.i.i = add nuw nsw i64 %conv.i.i.i.i.i.i, 1
  %conv1.i.i.i.i.i.i = zext i16 %globals.val.i.i.i.i.i to i64
  %mul.i.i.i.i.i.i = shl nuw nsw i64 %conv1.i.i.i.i.i.i, 1
  %mul3.i.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i.i.i, %add.i.i.i.i.i.i
  %add4.i.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i.i, 32
  %conv7.i.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i, %conv7.i.i.i.i.i
  %add9.i.i.i.i.i = add i64 %mul8.i.i.i.i.i, %cond.i.i.i.i.i.i.i
  %56 = inttoptr i64 %add9.i.i.i.i.i to ptr
  %conv.i.i.i.i.i = and i32 %43, 65535
  %clockBufferHead.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %56, i64 20
  %bf.load.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i, align 4
  %bf.lshr.i.i.i.i.i = lshr i32 %bf.load.i.i.i.i.i, 1
  %cmp3.not.i.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i.i, %conv.i.i.i.i.i
  br i1 %cmp3.not.i.i.i.i.i, label %if.then4.i.i.i.i.i, label %do.end9.i.i.i.i.i

if.then4.i.i.i.i.i:                               ; preds = %if.end.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i38.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i.i, align 4
  %.pre42.i.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i.i, 1
  br label %do.end9.i.i.i.i.i

do.end9.i.i.i.i.i:                                ; preds = %if.then4.i.i.i.i.i, %if.end.i.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i.i, %if.end.i.i.i.i ], [ %.pre42.i.i.i.i.i, %if.then4.i.i.i.i.i ]
  %and.i.i.i.i.i.i = and i64 %add9.i.i.i.i.i, -1073741824
  %57 = inttoptr i64 %and.i.i.i.i.i.i to ptr
  %sub.i.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i.i, %conv.i.i.i.i.i
  %clockBufferSize.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %57, i64 26
  %58 = load i16, ptr %clockBufferSize.i.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i.i = zext i16 %58 to i32
  %cmp17.i.i.i.i.i = icmp slt i32 %sub.i.i.i.i.i, %conv16.i.i.i.i.i
  br i1 %cmp17.i.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i, label %if.then18.i.i.i.i.i

if.then18.i.i.i.i.i:                              ; preds = %do.end9.i.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i38.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i: ; preds = %if.then18.i.i.i.i.i, %do.end9.i.i.i.i.i
  %59 = phi i16 [ %.pre.i.i.i.i.i, %if.then18.i.i.i.i.i ], [ %58, %do.end9.i.i.i.i.i ]
  %60 = urem i16 %write.sroa.0.0.extract.trunc.i, %59
  %rem.i.i.i.i.i = zext i16 %60 to i64
  %vectorClock.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %56, i64 30
  %numThreads.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %57, i64 24
  %61 = load i16, ptr %numThreads.i.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i.i = zext i16 %61 to i64
  %add.ptr.i.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i.i
  %mul.i8.i.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i.i, %rem.i.i.i.i.i
  %add.ptr.i.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i.i.i, i64 %mul.i8.i.i.i.i
  br label %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i

_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i: ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i, %do.body.i.i
  %retval.0.i.i22.i.i = phi ptr [ %add.ptr.i.i.i.i.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i.i ], [ null, %do.body.i.i ]
  %tobool.not.not.i.i.i = icmp eq ptr %retval.0.i.i22.i.i, null
  br i1 %tobool.not.not.i.i.i, label %cleanup.i.i.i, label %if.then1.i.i.i

if.then1.i.i.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i
  %62 = load i16, ptr %34, align 8, !tbaa !11
  %conv.i.i.i.i = zext i16 %62 to i32
  %cmp.not11.i.i.i.i = icmp eq i16 %62, 0
  br i1 %cmp.not11.i.i.i.i, label %cleanup.i.i.i, label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %if.then1.i.i.i, %for.body.i.i.i.i
  %i.012.i.i.i.i = phi i32 [ %inc.i.i.i.i, %for.body.i.i.i.i ], [ 0, %if.then1.i.i.i ]
  %idxprom.i.i.i.i = zext nneg i32 %i.012.i.i.i.i to i64
  %arrayidx.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idxprom.i.i.i.i
  %63 = load i16, ptr %arrayidx.i.i.i.i, align 2, !tbaa !22
  %arrayidx3.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %retval.0.i.i22.i.i, i64 %idxprom.i.i.i.i
  %64 = load i16, ptr %arrayidx3.i.i.i.i, align 2, !tbaa !22
  %cmp5.i.i.i.i = icmp uge i16 %63, %64
  %inc.i.i.i.i = add nuw nsw i32 %i.012.i.i.i.i, 1
  %exitcond.not.i.i.i.i = icmp ne i32 %inc.i.i.i.i, %conv.i.i.i.i
  %or.cond.not.i.i.i.i = select i1 %cmp5.i.i.i.i, i1 %exitcond.not.i.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i.i, label %for.body.i.i.i.i, label %cleanup.i.i.i, !llvm.loop !31

cleanup.i.i.i:                                    ; preds = %for.body.i.i.i.i, %if.then1.i.i.i, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i
  %retval.0.i23.i.i = phi i1 [ undef, %_ZN4gsan12_GLOBAL__N_119getSnapshotForWriteEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i.i ], [ true, %if.then1.i.i.i ], [ %cmp5.i.i.i.i, %for.body.i.i.i.i ]
  br i1 %tobool.not.not.i.i.i, label %cleanup.cont.i.i.i, label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i

cleanup.cont.i.i.i:                               ; preds = %cleanup.i.i.i
  %idxprom.i.i.i = zext nneg i16 %bf.clear3.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idxprom.i.i.i
  %65 = load i16, ptr %arrayidx.i.i.i, align 2, !tbaa !22
  %cmp7.i.i.i = icmp uge i16 %65, %write.sroa.0.0.extract.trunc.i
  br label %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i: ; preds = %cleanup.cont.i.i.i, %cleanup.i.i.i
  %retval.1.i.i.i = phi i1 [ %retval.0.i23.i.i, %cleanup.i.i.i ], [ %cmp7.i.i.i, %cleanup.cont.i.i.i ]
  br i1 %retval.1.i.i.i, label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i, label %if.then9.i.i

if.then9.i.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str1, ptr noundef nonnull %cond.i38.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i

_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i: ; preds = %if.then9.i.i, %_ZN4gsan12_GLOBAL__N_118clockHappensBeforeEPNS_11ThreadStateERKNS_11ScalarClockENS_8LocationE.exit.i.i, %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i.i, %land.rhs.i.i.i, %for.body.i
  %numReads1.i.i = getelementptr inbounds nuw i8, ptr %42, i64 20
  %66 = load i16, ptr %numReads1.i.i, align 4, !tbaa !32
  %conv.i59.i = zext i16 %66 to i32
  %cmp.not.i.i = icmp eq i16 %66, -1
  br i1 %cmp.not.i.i, label %if.end.i62.i, label %if.then.i60.i

if.then.i60.i:                                    ; preds = %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i
  %inc.i61.i = add nuw i16 %66, 1
  store i16 %inc.i61.i, ptr %numReads1.i.i, align 4, !tbaa !32
  br label %if.end.i62.i

if.end.i62.i:                                     ; preds = %if.then.i60.i, %_ZN4gsan12_GLOBAL__N_125assertOrderedOrCompatibleEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationEPKc.exit.i
  %67 = load i16, ptr %threadId.i56.i, align 4, !tbaa !22
  %idxprom.i.i64.i = zext i16 %67 to i64
  %arrayidx.i.i65.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i, i64 %idxprom.i.i64.i
  %68 = load i16, ptr %arrayidx.i.i65.i, align 2, !tbaa !22
  %bf.value.i.i.i = and i16 %67, 4095
  %bf.set6.i.i.i = or disjoint i16 %bf.value.i.i.i, %37
  %readClock.sroa.0.0.copyload.i.i = load i16, ptr %42, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.i.i = getelementptr inbounds nuw i8, ptr %42, i64 2
  %readClock.sroa.4.0.copyload.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.i.i, align 2, !tbaa !23
  %bf.clear.i.i = and i16 %readClock.sroa.4.0.copyload.i.i, 4095
  %cmp7.i66.i = icmp ne i16 %bf.clear.i.i, %67
  %cmp9.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.i.i, 0
  %or.cond.not.i.i = select i1 %cmp7.i66.i, i1 %cmp9.i.i, i1 false
  br i1 %or.cond.not.i.i, label %for.cond.i.i, label %if.then10.i.i

for.cond.i.i:                                     ; preds = %if.end.i62.i
  %arrayidx.1.i.i = getelementptr inbounds nuw i8, ptr %42, i64 4
  %readClock.sroa.0.0.copyload.1.i.i = load i16, ptr %arrayidx.1.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.1.i.i = getelementptr inbounds nuw i8, ptr %42, i64 6
  %readClock.sroa.4.0.copyload.1.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.1.i.i, align 2, !tbaa !23
  %bf.clear.1.i.i = and i16 %readClock.sroa.4.0.copyload.1.i.i, 4095
  %cmp7.1.i.i = icmp ne i16 %bf.clear.1.i.i, %67
  %cmp9.1.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.1.i.i, 0
  %or.cond.not.1.i.i = select i1 %cmp7.1.i.i, i1 %cmp9.1.i.i, i1 false
  br i1 %or.cond.not.1.i.i, label %for.cond.1.i.i, label %if.then10.i.i

for.cond.1.i.i:                                   ; preds = %for.cond.i.i
  %arrayidx.2.i.i = getelementptr inbounds nuw i8, ptr %42, i64 8
  %readClock.sroa.0.0.copyload.2.i.i = load i16, ptr %arrayidx.2.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.2.i.i = getelementptr inbounds nuw i8, ptr %42, i64 10
  %readClock.sroa.4.0.copyload.2.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.2.i.i, align 2, !tbaa !23
  %bf.clear.2.i.i = and i16 %readClock.sroa.4.0.copyload.2.i.i, 4095
  %cmp7.2.i.i = icmp ne i16 %bf.clear.2.i.i, %67
  %cmp9.2.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.2.i.i, 0
  %or.cond.not.2.i.i = select i1 %cmp7.2.i.i, i1 %cmp9.2.i.i, i1 false
  br i1 %or.cond.not.2.i.i, label %for.cond.2.i.i, label %if.then10.i.i

for.cond.2.i.i:                                   ; preds = %for.cond.1.i.i
  %arrayidx.3.i.i = getelementptr inbounds nuw i8, ptr %42, i64 12
  %readClock.sroa.0.0.copyload.3.i.i = load i16, ptr %arrayidx.3.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx.sroa_idx.3.i.i = getelementptr inbounds nuw i8, ptr %42, i64 14
  %readClock.sroa.4.0.copyload.3.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.3.i.i, align 2, !tbaa !23
  %bf.clear.3.i.i = and i16 %readClock.sroa.4.0.copyload.3.i.i, 4095
  %cmp7.3.i.i = icmp ne i16 %bf.clear.3.i.i, %67
  %cmp9.3.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.3.i.i, 0
  %or.cond.not.3.i.i = select i1 %cmp7.3.i.i, i1 %cmp9.3.i.i, i1 false
  br i1 %or.cond.not.3.i.i, label %for.cond.3.i.i, label %if.then10.i.i

for.cond.3.i.i:                                   ; preds = %for.cond.2.i.i
  %69 = atomicrmw add ptr %numReads18.i.i, i32 1 syncscope("block") monotonic, align 8
  %70 = load i32, ptr %rngSeed.i.i, align 16, !tbaa !34
  %conv21.i.i = zext i16 %67 to i32
  %mul.i.i.i.i68.i = mul i32 %69, -862048943
  %or.i.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i.i.i68.i, i32 %mul.i.i.i.i68.i, i32 15)
  %mul1.i.i.i.i.i = mul i32 %or.i.i.i.i.i.i, 461845907
  %xor.i.i.i.i = xor i32 %mul1.i.i.i.i.i, %70
  %or.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i.i.i.i, i32 %xor.i.i.i.i, i32 13)
  %mul.i.i.i.i = mul i32 %or.i.i.i.i.i, 5
  %add.i.i.i.i = add i32 %mul.i.i.i.i, -430675100
  %mul.i.i6.i.i.i = mul i32 %conv21.i.i, -862048943
  %or.i.i.i7.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i6.i.i.i, i32 %mul.i.i6.i.i.i, i32 15)
  %mul1.i.i8.i.i.i = mul i32 %or.i.i.i7.i.i.i, 461845907
  %xor.i9.i.i.i = xor i32 %add.i.i.i.i, %mul1.i.i8.i.i.i
  %or.i.i10.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i9.i.i.i, i32 %xor.i9.i.i.i, i32 13)
  %mul.i11.i.i.i = mul i32 %or.i.i10.i.i.i, 5
  %add.i12.i.i.i = add i32 %mul.i11.i.i.i, -430675100
  %shr.i.i.i.i = lshr i32 %add.i12.i.i.i, 16
  %71 = xor i32 %add.i12.i.i.i, %shr.i.i.i.i
  %xor.i13.i.i.i = xor i32 %71, 8
  %mul.i14.i.i.i = mul i32 %xor.i13.i.i.i, -2048144789
  %shr1.i.i.i.i = lshr i32 %mul.i14.i.i.i, 13
  %xor2.i.i.i.i = xor i32 %shr1.i.i.i.i, %mul.i14.i.i.i
  %mul3.i.i.i.i = mul i32 %xor2.i.i.i.i, -1028477387
  %shr4.i.i.i.i = lshr i32 %mul3.i.i.i.i, 16
  %xor5.i.i.i.i = xor i32 %shr4.i.i.i.i, %mul3.i.i.i.i
  %rem.i.i = urem i32 %xor5.i.i.i.i, %conv.i59.i
  %cmp24.i.i = icmp samesign ult i32 %rem.i.i, 4
  br i1 %cmp24.i.i, label %if.then25.i.i, label %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i

if.then10.i.i:                                    ; preds = %for.cond.2.i.i, %for.cond.1.i.i, %for.cond.i.i, %if.end.i62.i
  %arrayidx.lcssa.i.i = phi ptr [ %42, %if.end.i62.i ], [ %arrayidx.1.i.i, %for.cond.i.i ], [ %arrayidx.2.i.i, %for.cond.1.i.i ], [ %arrayidx.3.i.i, %for.cond.2.i.i ]
  %readClock.sroa.4.0.arrayidx.sroa_idx.le.i.i = getelementptr inbounds nuw i8, ptr %arrayidx.lcssa.i.i, i64 2
  store i16 %68, ptr %arrayidx.lcssa.i.i, align 4, !tbaa !22
  store i16 %bf.set6.i.i.i, ptr %readClock.sroa.4.0.arrayidx.sroa_idx.le.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i

if.then25.i.i:                                    ; preds = %for.cond.3.i.i
  %idxprom27.i.i = zext nneg i32 %rem.i.i to i64
  %arrayidx28.i.i = getelementptr inbounds nuw [4 x i8], ptr %42, i64 %idxprom27.i.i
  %scalarClock.sroa.5.0.arrayidx28.sroa_idx.i.i = getelementptr inbounds nuw i8, ptr %arrayidx28.i.i, i64 2
  store i16 %68, ptr %arrayidx28.i.i, align 4, !tbaa !22
  store i16 %bf.set6.i.i.i, ptr %scalarClock.sroa.5.0.arrayidx28.sroa_idx.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i

_ZN4gsan12_GLOBAL__N_110recordReadEPNS_11ThreadStateEPNS_10ShadowCellENS_11AtomicScopeE.exit.i: ; preds = %if.then25.i.i, %if.then10.i.i, %for.cond.3.i.i
  %inc.i = add nuw i8 %i.0114.i, 1
  %72 = load i8, ptr %numCells.i, align 8, !tbaa !48
  %cmp6.i = icmp ult i8 %inc.i, %72
  br i1 %cmp6.i, label %for.body.i, label %for.cond.cleanup.loopexit.i, !llvm.loop !52

for.body17.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, %for.body17.lr.ph.i
  %i10.0116.i = phi i8 [ 0, %for.body17.lr.ph.i ], [ %inc25.i, %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i ]
  %idxprom20.i = zext i8 %i10.0116.i to i64
  %arrayidx21.i = getelementptr inbounds nuw [8 x i8], ptr %cells19.i, i64 %idxprom20.i
  %73 = load ptr, ptr %arrayidx21.i, align 8, !tbaa !49
  %writeClock22.i = getelementptr inbounds nuw i8, ptr %73, i64 16
  %74 = load i32, ptr %writeClock22.i, align 4
  %write18.sroa.0.0.extract.trunc.i = trunc i32 %74 to i16
  %write18.sroa.4.0.extract.shift.i = lshr i32 %74, 16
  %write18.sroa.4.0.extract.trunc.i = trunc nuw i32 %write18.sroa.4.0.extract.shift.i to i16
  %75 = and i16 %write18.sroa.4.0.extract.trunc.i, 16384
  %bf.cast.not.i.i = icmp eq i16 %75, 0
  br i1 %bf.cast.not.i.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, label %if.end.i70.i

if.end.i70.i:                                     ; preds = %for.body17.i
  %76 = load i16, ptr %threadId.i71.i, align 4, !tbaa !22
  %bf.lshr2.i.i = lshr i16 %write18.sroa.4.0.extract.trunc.i, 12
  %77 = trunc nuw nsw i16 %bf.lshr2.i.i to i8
  %bf.cast4.i.i = and i8 %77, 3
  %bf.clear7.i.i = and i16 %write18.sroa.4.0.extract.trunc.i, 4095
  %cmp.i9.i.not.i73.i = icmp eq i8 %bf.cast4.i.i, 0
  br i1 %cmp.i9.i.not.i73.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, label %if.end.i.i74.i

if.end.i.i74.i:                                   ; preds = %if.end.i70.i
  switch i8 %switch.masked, label %default.unreachable [
    i8 1, label %sw.bb.i.i.i109.i
    i8 2, label %sw.bb2.i.i.i104.i
    i8 3, label %land.rhs.i.i75.i
  ]

sw.bb.i.i.i109.i:                                 ; preds = %if.end.i.i74.i
  %cmp.i10.i.i110.i = icmp eq i16 %76, %bf.clear7.i.i
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i

sw.bb2.i.i.i104.i:                                ; preds = %if.end.i.i74.i
  %78 = load i16, ptr %numSms.i.i.i105.i, align 4, !tbaa !21
  %79 = udiv i16 %76, %78
  %80 = udiv i16 %bf.clear7.i.i, %78
  %cmp9.i.i.i106.i = icmp eq i16 %79, %80
  br label %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i

_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i: ; preds = %sw.bb2.i.i.i104.i, %sw.bb.i.i.i109.i
  %retval.0.i.i.i108.i = phi i1 [ %cmp9.i.i.i106.i, %sw.bb2.i.i.i104.i ], [ %cmp.i10.i.i110.i, %sw.bb.i.i.i109.i ]
  br i1 %retval.0.i.i.i108.i, label %land.rhs.i.i75.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

land.rhs.i.i75.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i, %if.end.i.i74.i
  switch i8 %bf.cast4.i.i, label %default.unreachable [
    i8 1, label %sw.bb.i16.i.i101.i
    i8 2, label %sw.bb2.i13.i.i96.i
    i8 3, label %if.end10.i.i
  ]

sw.bb.i16.i.i101.i:                               ; preds = %land.rhs.i.i75.i
  %cmp.i17.i.i102.i = icmp eq i16 %76, %bf.clear7.i.i
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i

sw.bb2.i13.i.i96.i:                               ; preds = %land.rhs.i.i75.i
  %81 = load i16, ptr %numSms.i.i.i105.i, align 4, !tbaa !21
  %82 = udiv i16 %76, %81
  %83 = udiv i16 %bf.clear7.i.i, %81
  %cmp9.i15.i.i98.i = icmp eq i16 %82, %83
  br label %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i

_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i: ; preds = %sw.bb2.i13.i.i96.i, %sw.bb.i16.i.i101.i
  %retval.0.i.i100.i = phi i1 [ %cmp.i17.i.i102.i, %sw.bb.i16.i.i101.i ], [ %cmp9.i15.i.i98.i, %sw.bb2.i13.i.i96.i ]
  br i1 %retval.0.i.i100.i, label %if.end10.i.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

if.end10.i.i:                                     ; preds = %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i, %land.rhs.i.i75.i
  %84 = load i16, ptr %numSms.i.i.i105.i, align 4, !tbaa !21
  %bf.clear7.i.i.frozen = freeze i16 %bf.clear7.i.i
  %.frozen21 = freeze i16 %84
  %div16.i.i.i.i = udiv i16 %bf.clear7.i.i.frozen, %.frozen21
  %85 = mul i16 %div16.i.i.i.i, %.frozen21
  %rem17.i.i.i.i.decomposed = sub i16 %bf.clear7.i.i.frozen, %85
  %86 = load i64, ptr %globalsBase.i.i.i.i, align 8, !tbaa !20
  %conv5.i.i.i.i = zext nneg i16 %div16.i.i.i.i to i64
  %mul.i.i.i79.i = shl nuw nsw i64 %conv5.i.i.i.i, 30
  %add.i.i.i80.i = or disjoint i64 %mul.i.i.i79.i, 39
  %ptr.biased.i.i.i.i.i.i = add i64 %add.i.i.i80.i, %86
  %cond.i.i.i.i.i.i = and i64 %ptr.biased.i.i.i.i.i.i, -8
  %globals.val.i.i.i.i = load i16, ptr %40, align 8, !tbaa !11
  %globals.val15.i.i.i.i = load i16, ptr %41, align 2, !tbaa !15
  %conv.i.i.i.i81.i = zext i16 %globals.val15.i.i.i.i to i64
  %add.i.i.i.i82.i = add nuw nsw i64 %conv.i.i.i.i81.i, 1
  %conv1.i.i.i.i.i = zext i16 %globals.val.i.i.i.i to i64
  %mul.i.i.i.i83.i = shl nuw nsw i64 %conv1.i.i.i.i.i, 1
  %mul3.i.i.i.i.i = mul nuw nsw i64 %mul.i.i.i.i83.i, %add.i.i.i.i82.i
  %add4.i.i.i.i.i = add nuw nsw i64 %mul3.i.i.i.i.i, 32
  %conv7.i.i.i.i = zext nneg i16 %rem17.i.i.i.i.decomposed to i64
  %mul8.i.i.i.i = mul nuw nsw i64 %add4.i.i.i.i.i, %conv7.i.i.i.i
  %add9.i.i.i.i = add i64 %mul8.i.i.i.i, %cond.i.i.i.i.i.i
  %87 = inttoptr i64 %add9.i.i.i.i to ptr
  %conv.i.i.i84.i = and i32 %74, 65535
  %cmp.not.i.i.i.i = icmp eq i16 %write18.sroa.0.0.extract.trunc.i, 0
  br i1 %cmp.not.i.i.i.i, label %if.then.i.i.i.i, label %do.body1.i.i.i.i

if.then.i.i.i.i:                                  ; preds = %if.end10.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str3, ptr noundef nonnull %cond.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  br label %do.body1.i.i.i.i

do.body1.i.i.i.i:                                 ; preds = %if.then.i.i.i.i, %if.end10.i.i
  %clockBufferHead.i.i.i.i = getelementptr inbounds nuw i8, ptr %87, i64 20
  %bf.load.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i, align 4
  %bf.lshr.i.i.i.i = lshr i32 %bf.load.i.i.i.i, 1
  %cmp3.not.i.i.i.i = icmp samesign ult i32 %bf.lshr.i.i.i.i, %conv.i.i.i84.i
  br i1 %cmp3.not.i.i.i.i, label %if.then4.i.i.i.i, label %do.end9.i.i.i.i

if.then4.i.i.i.i:                                 ; preds = %do.body1.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %bf.load13.pre.i.i.i.i = load i32, ptr %clockBufferHead.i.i.i.i, align 4
  %.pre42.i.i.i.i = lshr i32 %bf.load13.pre.i.i.i.i, 1
  br label %do.end9.i.i.i.i

do.end9.i.i.i.i:                                  ; preds = %if.then4.i.i.i.i, %do.body1.i.i.i.i
  %bf.lshr14.pre-phi.i.i.i.i = phi i32 [ %bf.lshr.i.i.i.i, %do.body1.i.i.i.i ], [ %.pre42.i.i.i.i, %if.then4.i.i.i.i ]
  %and.i.i.i.i85.i = and i64 %add9.i.i.i.i, -1073741824
  %88 = inttoptr i64 %and.i.i.i.i85.i to ptr
  %sub.i.i.i.i = sub nsw i32 %bf.lshr14.pre-phi.i.i.i.i, %conv.i.i.i84.i
  %clockBufferSize.i.i.i.i = getelementptr inbounds nuw i8, ptr %88, i64 26
  %89 = load i16, ptr %clockBufferSize.i.i.i.i, align 2, !tbaa !15
  %conv16.i.i.i.i = zext i16 %89 to i32
  %cmp17.i.i.i.i = icmp slt i32 %sub.i.i.i.i, %conv16.i.i.i.i
  br i1 %cmp17.i.i.i.i, label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, label %if.then18.i.i.i.i

if.then18.i.i.i.i:                                ; preds = %do.end9.i.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #8
  %.pre.i.i.i.i = load i16, ptr %clockBufferSize.i.i.i.i, align 2, !tbaa !15
  br label %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i

_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i: ; preds = %if.then18.i.i.i.i, %do.end9.i.i.i.i
  %90 = phi i16 [ %.pre.i.i.i.i, %if.then18.i.i.i.i ], [ %89, %do.end9.i.i.i.i ]
  %91 = urem i16 %write18.sroa.0.0.extract.trunc.i, %90
  %rem.i.i.i.i = zext i16 %91 to i64
  %vectorClock.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %87, i64 30
  %numThreads.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %88, i64 24
  %92 = load i16, ptr %numThreads.i.i.i.i.i, align 8, !tbaa !11
  %idx.ext.i.i.i.i.i = zext i16 %92 to i64
  %add.ptr.i.i.i.i86.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i.i.i, i64 %idx.ext.i.i.i.i.i
  %mul.i8.i.i.i = mul nuw nsw i64 %idx.ext.i.i.i.i.i, %rem.i.i.i.i
  %add.ptr.i.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i86.i, i64 %mul.i8.i.i.i
  %93 = load i16, ptr %40, align 8, !tbaa !11
  %cmp2.not.i.i = icmp eq i16 %93, 0
  br i1 %cmp2.not.i.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, label %for.body.i87.i

for.cond.cleanup.i93.i:                           ; preds = %for.inc.i.i
  br i1 %changed.1.off0.i.i, label %if.then25.i94.i, label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

for.body.i87.i:                                   ; preds = %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, %for.inc.i.i
  %94 = phi i16 [ %97, %for.inc.i.i ], [ %93, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ]
  %i.04.i.i = phi i32 [ %inc.i90.i, %for.inc.i.i ], [ 0, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ]
  %changed.0.off03.i.i = phi i1 [ %changed.1.off0.i.i, %for.inc.i.i ], [ false, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i ]
  %idxprom.i88.i = zext nneg i32 %i.04.i.i to i64
  %arrayidx.i89.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i, i64 %idxprom.i88.i
  %95 = load i16, ptr %arrayidx.i89.i, align 2, !tbaa !22
  %arrayidx15.i.i = getelementptr inbounds nuw [2 x i8], ptr %add.ptr.i.i.i.i, i64 %idxprom.i88.i
  %96 = load i16, ptr %arrayidx15.i.i, align 2, !tbaa !22
  %cmp17.i.i = icmp ult i16 %95, %96
  br i1 %cmp17.i.i, label %if.then18.i.i, label %for.inc.i.i

if.then18.i.i:                                    ; preds = %for.body.i87.i
  store i16 %96, ptr %arrayidx.i89.i, align 2, !tbaa !22
  %.pre.i.i = load i16, ptr %40, align 8, !tbaa !11
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %if.then18.i.i, %for.body.i87.i
  %97 = phi i16 [ %.pre.i.i, %if.then18.i.i ], [ %94, %for.body.i87.i ]
  %changed.1.off0.i.i = phi i1 [ true, %if.then18.i.i ], [ %changed.0.off03.i.i, %for.body.i87.i ]
  %inc.i90.i = add nuw nsw i32 %i.04.i.i, 1
  %conv.i91.i = zext i16 %97 to i32
  %cmp.i92.i = icmp samesign ult i32 %inc.i90.i, %conv.i91.i
  br i1 %cmp.i92.i, label %for.body.i87.i, label %for.cond.cleanup.i93.i, !llvm.loop !53

if.then25.i94.i:                                  ; preds = %for.cond.cleanup.i93.i
  %bf.load26.i.i = load i32, ptr %clockBufferDirty.i95.i, align 4
  %bf.set.i.i = or i32 %bf.load26.i.i, 1
  store i32 %bf.set.i.i, ptr %clockBufferDirty.i95.i, align 4
  br label %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i

_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i: ; preds = %if.then25.i94.i, %for.cond.cleanup.i93.i, %_ZN4gsan12_GLOBAL__N_118getClockBufferSlotEPNS_11ThreadStateEtNS_8LocationE.exit.i.i.i, %_ZN4gsan12_GLOBAL__N_125areAtomicScopesCompatibleENS_11AtomicScopeEtS1_tPNS_11GlobalStateE.exit.i99.i, %_ZN4gsan12_GLOBAL__N_115scopeCoversPairENS_11AtomicScopeEttPNS_11GlobalStateE.exit.i.i107.i, %if.end.i70.i, %for.body17.i
  %inc25.i = add nuw i8 %i10.0116.i, 1
  %98 = load i8, ptr %numCells.i, align 8, !tbaa !48
  %cmp15.i = icmp ult i8 %inc25.i, %98
  br i1 %cmp15.i, label %for.body17.i, label %_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit, !llvm.loop !54

_ZN4gsan12_GLOBAL__N_117beginAtomicAccessEPNS_11GlobalStateEPNS_16AtomicEventStateEbmijjNS_8LocationE.exit: ; preds = %_ZN4gsan12_GLOBAL__N_117maybeMergeAcquireEPNS_11ThreadStateENS_11AtomicScopeERKNS_11ScalarClockENS_8LocationE.exit.i, %entry, %_ZN4gsan12_GLOBAL__N_124acquireAtomicShadowRangeEPNS_11ThreadStateEPNS_16AtomicEventStateEmiNS_8LocationE.exit.i, %for.cond.cleanup.i, %for.cond11.preheader.i
  ret void
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_atomic_end_scalar(ptr noundef captures(none) %eventState, i32 noundef %pred, i32 noundef %didWrite, i32 noundef %sem, i32 noundef %scope, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %agg.tmp = alloca %"struct.gsan::Location", align 8
  %cmp = icmp ne i32 %pred, 0
  %cmp3 = icmp ne i32 %didWrite, 0
  store ptr %file, ptr %agg.tmp, align 8, !tbaa !55
  %loc.sroa.4.0.agg.tmp.sroa_idx = getelementptr inbounds nuw i8, ptr %agg.tmp, i64 8
  store i32 %line, ptr %loc.sroa.4.0.agg.tmp.sroa_idx, align 8, !tbaa !3
  tail call fastcc void @_ZN4gsan12_GLOBAL__N_115endAtomicAccessEPNS_16AtomicEventStateEbbjjNS_8LocationE(ptr noundef %eventState, i1 noundef zeroext %cmp, i1 noundef zeroext %cmp3, i32 noundef %sem, i32 noundef %scope, ptr noundef nonnull byval(%"struct.gsan::Location") align 8 %agg.tmp) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.smid() #3

; Function Attrs: convergent nounwind denormal_fpenv(float: preservesign)
declare dso_local void @__assertfail(ptr noundef, ptr noundef, i32 noundef, ptr noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #6

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.vector.reduce.add.v2i8(<2 x i8>) #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn denormal_fpenv(float: preservesign) memory(argmem: read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+sm_80" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress nounwind denormal_fpenv(float: preservesign) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+sm_80" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nounwind denormal_fpenv(float: preservesign) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+sm_80" "uniform-work-group-size" }
attributes #5 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #8 = { convergent nounwind "uniform-work-group-size" }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!llvm.errno.tbaa = !{!3}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project 87717bf9f81f7b29466c5d9a30a3453bdfc93941)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !9, i64 0}
!8 = !{!"_ZTSN4gsan8LocationE", !9, i64 0, !4, i64 8}
!9 = !{!"p1 omnipotent char", !10, i64 0}
!10 = !{!"any pointer", !5, i64 0}
!11 = !{!12, !14, i64 24}
!12 = !{!"_ZTSN4gsan11GlobalStateE", !13, i64 0, !13, i64 8, !4, i64 16, !14, i64 20, !14, i64 22, !14, i64 24, !14, i64 26}
!13 = !{!"long", !5, i64 0}
!14 = !{!"short", !5, i64 0}
!15 = !{!12, !14, i64 26}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 _ZTSN4gsan11GlobalStateE", !10, i64 0}
!18 = !{!12, !13, i64 0}
!19 = !{!13, !13, i64 0}
!20 = !{!12, !13, i64 8}
!21 = !{!12, !14, i64 20}
!22 = !{!14, !14, i64 0}
!23 = !{!5, !5, i64 0}
!24 = distinct !{!24, !25}
!25 = !{!"llvm.loop.mustprogress"}
!26 = distinct !{!26, !25}
!27 = !{!28, !14, i64 0}
!28 = !{!"_ZTSN4gsan11ScalarClockE", !14, i64 0, !14, i64 2, !29, i64 3, !30, i64 3}
!29 = !{!"_ZTSN4gsan11AtomicScopeE", !5, i64 0}
!30 = !{!"bool", !5, i64 0}
!31 = distinct !{!31, !25}
!32 = !{!33, !14, i64 20}
!33 = !{!"_ZTSN4gsan10ShadowCellE", !5, i64 0, !28, i64 16, !14, i64 20, !14, i64 22}
!34 = !{!12, !4, i64 16}
!35 = distinct !{!35, !25}
!36 = distinct !{!36, !25}
!37 = distinct !{!37, !25}
!38 = distinct !{!38, !25}
!39 = distinct !{!39, !25}
!40 = distinct !{!40, !25, !41, !42}
!41 = !{!"llvm.loop.isvectorized", i32 1}
!42 = !{!"llvm.loop.unroll.runtime.disable"}
!43 = distinct !{!43, !25, !42, !41}
!44 = distinct !{!44, !25}
!45 = !{!46, !47, i64 0}
!46 = !{!"_ZTSN4gsan16AtomicEventStateE", !47, i64 0, !5, i64 8, !5, i64 32}
!47 = !{!"p1 _ZTSN4gsan11ThreadStateE", !10, i64 0}
!48 = !{!46, !5, i64 32}
!49 = !{!50, !50, i64 0}
!50 = !{!"p1 _ZTSN4gsan10ShadowCellE", !10, i64 0}
!51 = distinct !{!51, !25}
!52 = distinct !{!52, !25}
!53 = distinct !{!53, !25}
!54 = distinct !{!54, !25}
!55 = !{!9, !9, i64 0}
!56 = distinct !{!56, !25}
!57 = distinct !{!57, !25}
!58 = distinct !{!58, !25}
!59 = distinct !{!59, !25}
!60 = distinct !{!60, !25}
!61 = distinct !{!61, !25}
!62 = distinct !{!62, !25, !41, !42}
!63 = distinct !{!63, !25, !42, !41}
