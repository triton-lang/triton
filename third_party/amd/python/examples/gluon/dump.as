(M=1024, N=1024, K=512), (BLOCK_M=256, BLOCK_N=256, BLOCK_K=128), TRANSPOSE_B=True, NUM_WARPS=8, NUM_BUFFERS=2, PERSISTENT=False, PREFETCH=False
// -----// AMDGCN Dump //----- //
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	gemm_tdm_pipelined_kernel       ; -- Begin function gemm_tdm_pipelined_kernel
	.p2align	8
	.type	gemm_tdm_pipelined_kernel,@function
gemm_tdm_pipelined_kernel:              ; @gemm_tdm_pipelined_kernel
.Lfunc_begin0:
	.file	1 "/home/jung/jp/triton/third_party/amd/python/examples/gluon" "f16_gemm_gfx1250.py"
	.loc	1 277 0                         ; f16_gemm_gfx1250.py:277:0
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s14, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
.Ltmp0:
	.loc	1 86 36 prologue_end            ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_and_b32_e32 v14, 0xe0, v0
	s_add_co_i32 s1, s1, s0
	s_cmp_eq_u32 s14, 0
	s_mov_b32 s22, 0
	s_cselect_b32 s1, ttmp9, s1
.Ltmp1:
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language" "standard.py"
	.loc	2 43 17                         ; standard.py:43:17 @[ f16_gemm_gfx1250.py:300:29 ]
	s_add_co_i32 s0, s8, 0xff
.Ltmp2:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_abs_i32 s19, s1
.Ltmp3:
	.loc	2 43 30                         ; standard.py:43:30 @[ f16_gemm_gfx1250.py:300:29 ]
	s_ashr_i32 s14, s0, 31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp4:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mul_lo_u32 v2 /*v258*/, s11, v14
.Ltmp5:
	.loc	2 43 30                         ; standard.py:43:30 @[ f16_gemm_gfx1250.py:300:29 ]
	s_lshr_b32 s14, s14, 24
.Ltmp6:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mul_lo_u32 v4 /*v260*/, s12, v14
.Ltmp7:
	.loc	2 43 30                         ; standard.py:43:30 @[ f16_gemm_gfx1250.py:300:29 ]
	s_add_co_i32 s0, s0, s14
	s_mov_b32 s20, 32
	s_ashr_i32 s14, s0, 8
.Ltmp8:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	s_mov_b32 s23, s22
.Ltmp9:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_abs_i32 s15, s14
	v_and_b32_e32 v78 /*v334*/, 16, v0
	s_cvt_f32_u32 s0, s15
	s_sub_co_i32 s18, 0, s15
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_2)
.Ltmp10:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_dual_ashrrev_i32 v3 /*v259*/, 31, v2 /*v258*/ :: v_dual_ashrrev_i32 v5 /*v261*/, 31, v4 /*v260*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp11:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	v_rcp_iflag_f32_e32 v1, s0
	s_mov_b32 s0, 1
.Ltmp12:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mov_b64_e32 v[4:5], s[2:3]
	v_mov_b64_e32 v[2:3], s[0:1]
	v_mov_b32_e32 v5, s11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
.Ltmp13:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	v_readfirstlane_b32 s16, v1
.Ltmp14:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mul_u32_u24_e32 v1, 0x88, v14
	v_readfirstlane_b32 s24, v2
	v_sub_nc_u32_e32 v2, s8, v14
.Ltmp15:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_mul_f32 s17, s16, 0x4f7ffffe
.Ltmp16:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_lshlrev_b32_e32 v15, 1, v1
	s_mov_b32 s16, 0x7510000
	v_readfirstlane_b32 s41, v5
.Ltmp17:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_cvt_u32_f32 s17, s17
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp18:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_dual_add_nc_u32 v79 /*v335*/, 0, v15 :: v_dual_max_i32 v6 /*v262*/, 0, v2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
.Ltmp19:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_mul_i32 s18, s18, s17
	s_mul_hi_u32 s18, s17, s18
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp20:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_lshrrev_b32_e32 v1, 16, v6 /*v262*/
.Ltmp21:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_add_co_i32 s17, s17, s18
	s_xor_b32 s18, s1, s14
	s_mul_hi_u32 s17, s19, s17
	s_ashr_i32 s18, s18, 31
	s_mul_i32 s21, s17, s15
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp22:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_or_b32_e32 v80 /*v336*/, 0x800000, v1
.Ltmp23:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_sub_co_i32 s19, s19, s21
	s_add_co_i32 s21, s17, 1
	s_sub_co_i32 s25, s19, s15
	s_cmp_ge_u32 s19, s15
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp24:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mov_b32_e32 v1, v79 /*v335*/
.Ltmp25:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_cselect_b32 s17, s21, s17
	s_cselect_b32 s19, s25, s19
	s_add_co_i32 s21, s17, 1
	s_cmp_ge_u32 s19, s15
.Ltmp26:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mov_b32_e32 v3, v80 /*v336*/
.Ltmp27:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_cselect_b32 s15, s21, s17
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp28:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_readfirstlane_b32 s25, v1
.Ltmp29:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_xor_b32 s15, s15, s18
.Ltmp30:
	.loc	1 77 12                         ; f16_gemm_gfx1250.py:77:12 @[ f16_gemm_gfx1250.py:306:100 ]
	v_alignbit_b32 v1, s9, s10, 16
.Ltmp31:
	.loc	1 302 19                        ; f16_gemm_gfx1250.py:302:19
	s_sub_co_i32 s18, s15, s18
.Ltmp32:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_readfirstlane_b32 s39, v3
.Ltmp33:
	.loc	1 301 18                        ; f16_gemm_gfx1250.py:301:18
	s_mul_i32 s14, s18, s14
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp34:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_add3_u32 v81 /*v337*/, 0x21ff0, 0, v15
.Ltmp35:
	.loc	1 301 18                        ; f16_gemm_gfx1250.py:301:18
	s_sub_co_i32 s1, s1, s14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	.loc	1 304 69                        ; f16_gemm_gfx1250.py:304:69
	s_lshl_b32 s1, s1, 8
	.loc	1 304 79 is_stmt 0              ; f16_gemm_gfx1250.py:304:79
	s_mul_i32 s14, s1, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
.Ltmp36:
	.loc	1 59 21 is_stmt 1               ; f16_gemm_gfx1250.py:59:21 @[ f16_gemm_gfx1250.py:306:100 ]
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[14:15], s[14:15], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[2:3], s[2:3], s[14:15]
.Ltmp37:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	s_max_i32 s14, s10, 0
	s_bitset0_b32 s3, 31
	s_lshl_b32 s17, s14, 16
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[12:13], v[2:3] /*v[258:259]*/, 1, s[2:3]
	v_mov_b64_e32 v[4:5], s[16:17]
	v_mov_b64_e32 v[8:9], s[20:21]
	v_mov_b64_e32 v[10:11], s[22:23]
	v_alignbit_b32 v2, v6 /*v262*/, s14, 16
	v_or_b32_e32 v3, 0x80000000, v13
.Ltmp38:
	.loc	1 304 98                        ; f16_gemm_gfx1250.py:304:98
	s_lshl_b32 s14, s18, 8
.Ltmp39:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_mov_b64_e32 v[6:7], s[18:19]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_readfirstlane_b32 s36, v4
	v_mov_b32_e32 v4, v12
	v_readfirstlane_b32 s37, v5
	v_readfirstlane_b32 s40, v8
	v_readfirstlane_b32 s42, v10
	v_readfirstlane_b32 s38, v2
	v_readfirstlane_b32 s27, v3
	v_readfirstlane_b32 s43, v11
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_lshrrev_b32_e32 v2, 16, v1
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	v_readfirstlane_b32 s26, v4
.Ltmp40:
	.loc	1 304 108                       ; f16_gemm_gfx1250.py:304:108
	s_mul_i32 s18, s14, s12
.Ltmp41:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	s_and_b32 s15, s9, 0xffff0000
.Ltmp42:
	.loc	1 73 25                         ; f16_gemm_gfx1250.py:73:25 @[ f16_gemm_gfx1250.py:306:100 ]
	s_ashr_i32 s19, s18, 31
.Ltmp43:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_sub_nc_u32_e32 v3, s15, v14
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	tensor_load_to_lds s[24:27], s[36:43]
.Ltmp44:
	.loc	1 73 25                         ; f16_gemm_gfx1250.py:73:25 @[ f16_gemm_gfx1250.py:306:100 ]
	s_lshl_b64 s[18:19], s[18:19], 1
.Ltmp45:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_readfirstlane_b32 s15, v1
.Ltmp46:
	.loc	1 73 25                         ; f16_gemm_gfx1250.py:73:25 @[ f16_gemm_gfx1250.py:306:100 ]
	s_add_nc_u64 s[4:5], s[4:5], s[18:19]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp47:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_add_max_i32_e64 v7 /*v263*/, v3, v2, 0
	s_and_b64 s[4:5], s[4:5], 0x7fffffffffffffff
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:315:105 ]
	s_and_b32 s18, s10, 0xffff
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	v_lshl_add_u64 v[2:3], v[4:5] /*v[260:261]*/, 1, s[4:5]
	s_lshl_b32 s19, s15, 16
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshrrev_b32_e32 v1, 16, v7 /*v263*/
	s_or_b32 s15, s19, s18
	v_mov_b64_e32 v[12:13], s[2:3]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v3, 0x80000000, v3
	s_max_i32 s15, s15, 0
	v_mov_b32_e32 v12, v2
	s_lshl_b32 s17, s15, 16
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_alignbit_b32 v14, v7 /*v263*/, s15, 16
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_or_b32_e32 v82 /*v338*/, 0x800000, v1
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[10:11], s[0:1]
	v_mov_b32_e32 v11, v3
	v_mov_b64_e32 v[2:3], s[16:17]
	v_mov_b64_e32 v[6:7], s[20:21]
	v_mov_b64_e32 v[8:9], s[22:23]
	v_mov_b64_e32 v[4:5], s[18:19]
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v1, v81 /*v337*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v4, v14
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v5, v82 /*v338*/ :: v_dual_mov_b32 v7, s12
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_readfirstlane_b32 s36, v10
	v_readfirstlane_b32 s37, v1
	v_readfirstlane_b32 s38, v12
	v_readfirstlane_b32 s39, v11
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_readfirstlane_b32 s28, v6
	v_readfirstlane_b32 s29, v7
	v_readfirstlane_b32 s30, v8
	v_readfirstlane_b32 s31, v9
	v_lshlrev_b32_e32 v5 /*v261*/, 7, v0
.Ltmp48:
	.loc	2 43 17                         ; standard.py:43:17 @[ f16_gemm_gfx1250.py:317:35 ]
	s_add_co_i32 s0, s10, 0x7f
	s_delay_alu instid0(SALU_CYCLE_1)
.Ltmp49:
	.loc	1 317 22                        ; f16_gemm_gfx1250.py:317:22
	s_cmp_gt_i32 s0, 0xff
.Ltmp50:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:315:105 ]
	tensor_load_to_lds s[36:39], s[24:31]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp51:
	.loc	1 317 22                        ; f16_gemm_gfx1250.py:317:22
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	.loc	1 0 22 is_stmt 0                ; f16_gemm_gfx1250.py:0:22
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp52:
	.loc	1 102 52 is_stmt 1              ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_and_b32_e32 v3 /*v259*/, 16, v0
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_bitop3_b32 v1, v5 /*v261*/, 0x1010, v0 bitop3:0xc8
	s_mov_b32 s15, 0
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_3
.Ltmp53:
.LBB0_2:
	.loc	1 0 72 is_stmt 0                ; f16_gemm_gfx1250.py:0:72
	s_mov_b32 s15, -1
                                        ; implicit-def: $vgpr259
                                        ; implicit-def: $vgpr1
.LBB0_3:                                ; %Flow
	v_mov_b32_e32 v9, 0
	s_and_not1_b32 vcc_lo, exec_lo, s15
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v24, v9 :: v_dual_mov_b32 v23, v9
	v_dual_mov_b32 v22, v9 :: v_dual_mov_b32 v21, v9
	v_dual_mov_b32 v20, v9 :: v_dual_mov_b32 v19, v9
	v_dual_mov_b32 v18, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v31, v9
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v27, v9
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v9 :: v_dual_mov_b32 v39, v9
	v_dual_mov_b32 v38, v9 :: v_dual_mov_b32 v37, v9
	v_dual_mov_b32 v36, v9 :: v_dual_mov_b32 v35, v9
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v88, v9 :: v_dual_mov_b32 v87, v9
	v_dual_mov_b32 v86, v9 :: v_dual_mov_b32 v85, v9
	v_dual_mov_b32 v84, v9 :: v_dual_mov_b32 v83, v9
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v105, v9
	v_dual_mov_b32 v104, v9 :: v_dual_mov_b32 v103, v9
	v_dual_mov_b32 v102, v9 :: v_dual_mov_b32 v101, v9
	v_dual_mov_b32 v100, v9 :: v_dual_mov_b32 v99, v9
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v121, v9
	v_dual_mov_b32 v120, v9 :: v_dual_mov_b32 v119, v9
	v_dual_mov_b32 v118, v9 :: v_dual_mov_b32 v117, v9
	v_dual_mov_b32 v116, v9 :: v_dual_mov_b32 v115, v9
	v_dual_mov_b32 v114, v9 :: v_dual_mov_b32 v129, v9
	v_dual_mov_b32 v128, v9 :: v_dual_mov_b32 v127, v9
	v_dual_mov_b32 v126, v9 :: v_dual_mov_b32 v125, v9
	v_dual_mov_b32 v124, v9 :: v_dual_mov_b32 v123, v9
	v_dual_mov_b32 v122, v9 :: v_dual_mov_b32 v137, v9
	v_dual_mov_b32 v136, v9 :: v_dual_mov_b32 v135, v9
	v_dual_mov_b32 v134, v9 :: v_dual_mov_b32 v133, v9
	v_dual_mov_b32 v132, v9 :: v_dual_mov_b32 v131, v9
	v_dual_mov_b32 v130, v9 :: v_dual_mov_b32 v145, v9
	v_dual_mov_b32 v144, v9 :: v_dual_mov_b32 v143, v9
	v_dual_mov_b32 v142, v9 :: v_dual_mov_b32 v141, v9
	v_dual_mov_b32 v140, v9 :: v_dual_mov_b32 v139, v9
	v_dual_mov_b32 v138, v9 :: v_dual_mov_b32 v153, v9
	v_dual_mov_b32 v152, v9 :: v_dual_mov_b32 v151, v9
	v_dual_mov_b32 v150, v9 :: v_dual_mov_b32 v149, v9
	v_dual_mov_b32 v148, v9 :: v_dual_mov_b32 v147, v9
	v_dual_mov_b32 v146, v9 :: v_dual_mov_b32 v161, v9
	v_dual_mov_b32 v160, v9 :: v_dual_mov_b32 v159, v9
	v_dual_mov_b32 v158, v9 :: v_dual_mov_b32 v157, v9
	v_dual_mov_b32 v156, v9 :: v_dual_mov_b32 v155, v9
	v_dual_mov_b32 v154, v9 :: v_dual_mov_b32 v169, v9
	v_dual_mov_b32 v168, v9 :: v_dual_mov_b32 v167, v9
	v_dual_mov_b32 v166, v9 :: v_dual_mov_b32 v165, v9
	v_dual_mov_b32 v164, v9 :: v_dual_mov_b32 v163, v9
	v_dual_mov_b32 v162, v9 :: v_dual_mov_b32 v177, v9
	v_dual_mov_b32 v176, v9 :: v_dual_mov_b32 v175, v9
	v_dual_mov_b32 v174, v9 :: v_dual_mov_b32 v173, v9
	v_dual_mov_b32 v172, v9 :: v_dual_mov_b32 v171, v9
	v_dual_mov_b32 v170, v9 :: v_dual_mov_b32 v185, v9
	v_dual_mov_b32 v184, v9 :: v_dual_mov_b32 v183, v9
	v_dual_mov_b32 v182, v9 :: v_dual_mov_b32 v181, v9
	v_dual_mov_b32 v180, v9 :: v_dual_mov_b32 v179, v9
	v_dual_mov_b32 v178, v9 :: v_dual_mov_b32 v193, v9
	v_dual_mov_b32 v192, v9 :: v_dual_mov_b32 v191, v9
	v_dual_mov_b32 v190, v9 :: v_dual_mov_b32 v189, v9
	v_dual_mov_b32 v188, v9 :: v_dual_mov_b32 v187, v9
	v_dual_mov_b32 v186, v9 :: v_dual_mov_b32 v201, v9
	v_dual_mov_b32 v200, v9 :: v_dual_mov_b32 v199, v9
	v_dual_mov_b32 v198, v9 :: v_dual_mov_b32 v197, v9
	v_dual_mov_b32 v196, v9 :: v_dual_mov_b32 v195, v9
	v_dual_mov_b32 v194, v9 :: v_dual_mov_b32 v209, v9
	v_dual_mov_b32 v208, v9 :: v_dual_mov_b32 v207, v9
	v_dual_mov_b32 v206, v9 :: v_dual_mov_b32 v205, v9
	v_dual_mov_b32 v204, v9 :: v_dual_mov_b32 v203, v9
	v_dual_mov_b32 v202, v9 :: v_dual_mov_b32 v217, v9
	v_dual_mov_b32 v216, v9 :: v_dual_mov_b32 v215, v9
	v_dual_mov_b32 v214, v9 :: v_dual_mov_b32 v213, v9
	v_dual_mov_b32 v212, v9 :: v_dual_mov_b32 v211, v9
	v_dual_mov_b32 v210, v9 :: v_dual_mov_b32 v225, v9
	v_dual_mov_b32 v224, v9 :: v_dual_mov_b32 v223, v9
	v_dual_mov_b32 v222, v9 :: v_dual_mov_b32 v221, v9
	v_dual_mov_b32 v220, v9 :: v_dual_mov_b32 v219, v9
	v_dual_mov_b32 v218, v9 :: v_dual_mov_b32 v233, v9
	v_dual_mov_b32 v232, v9 :: v_dual_mov_b32 v231, v9
	v_dual_mov_b32 v230, v9 :: v_dual_mov_b32 v229, v9
	v_dual_mov_b32 v228, v9 :: v_dual_mov_b32 v227, v9
	v_dual_mov_b32 v226, v9 :: v_dual_mov_b32 v241, v9
	v_dual_mov_b32 v240, v9 :: v_dual_mov_b32 v239, v9
	v_dual_mov_b32 v238, v9 :: v_dual_mov_b32 v237, v9
	v_dual_mov_b32 v236, v9 :: v_dual_mov_b32 v235, v9
	v_dual_mov_b32 v234, v9 :: v_dual_mov_b32 v249, v9
	v_dual_mov_b32 v248, v9 :: v_dual_mov_b32 v247, v9
	v_dual_mov_b32 v246, v9 :: v_dual_mov_b32 v245, v9
	v_dual_mov_b32 v244, v9 :: v_dual_mov_b32 v243, v9
	v_dual_mov_b32 v242, v9 :: v_dual_mov_b32 v255, v9
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v1 /*v257*/, v9 :: v_dual_mov_b32 v0 /*v256*/, v9
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v254, v9 :: v_dual_mov_b32 v253, v9
	v_dual_mov_b32 v252, v9 :: v_dual_mov_b32 v251, v9
	v_mov_b32_e32 v250, v9
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	v_dual_lshlrev_b32 v1, 8, v0 :: v_dual_lshlrev_b32 v2, 6, v0
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_lshlrev_b32 v83 /*v339*/, 16, v6 /*v262*/ :: v_dual_lshlrev_b32 v84 /*v340*/, 16, v7 /*v263*/
	.loc	1 317 22 is_stmt 1              ; f16_gemm_gfx1250.py:317:22
	v_add_nc_u32_e32 v76 /*v332*/, 0x80, v2 /*v258*/
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_and_b32_e32 v3, 0xf00, v1
	v_bitop3_b32 v1, v5 /*v261*/, 0x1010, v0 bitop3:0xc8
.Ltmp54:
	.loc	2 43 30                         ; standard.py:43:30 @[ f16_gemm_gfx1250.py:317:35 ]
	s_lshr_b32 s15, s0, 7
.Ltmp55:
	.loc	1 317 22                        ; f16_gemm_gfx1250.py:317:22
	s_add_co_i32 s0, s19, s18
	s_addk_co_i32 s10, 0xff80
	v_and_or_b32 v2, 0x3000, v2, v3
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v3, v1, v3
	s_add_co_i32 s18, s0, 0xffffff80
	s_mov_b32 s0, 1
	s_mov_b32 s23, s22
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_dual_mov_b32 v250, 0 :: v_dual_bitop2_b32 v4, v2, v78 /*v334*/ bitop3:0x54
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v5, 0x4000, v2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v85 /*v341*/, 4, v2
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v6, 0x8000, v2
	v_or_b32_e32 v2, 0xc000, v2
	v_or_b32_e32 v7, 0x2000, v3
	v_or_b32_e32 v8, 0x4000, v3
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v86 /*v342*/, 4, v5 :: v_dual_lshrrev_b32 v87 /*v343*/, 4, v6
	v_add_nc_u32_e32 v97 /*v353*/, 0, v4
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v5, 4, v7 :: v_dual_lshrrev_b32 v7, 4, v8
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v88 /*v344*/, 4, v2 :: v_dual_mov_b32 v0 /*v256*/, v250
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v2, 0x8000, v3
	v_or_b32_e32 v10, 0x6000, v3
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v91 /*v347*/, 0x5f0, v7
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v6, 0xc000, v3
	v_or_b32_e32 v7, 0xe000, v3
	v_lshrrev_b32_e32 v2, 4, v2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v90 /*v346*/, 0x3f0, v5
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v5, 0xa000, v3
	v_dual_lshrrev_b32 v9, 4, v3 :: v_dual_lshrrev_b32 v8, 4, v10
	v_dual_lshrrev_b32 v6, 4, v6 :: v_dual_lshrrev_b32 v7, 4, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v251, v250 :: v_dual_lshrrev_b32 v5, 4, v5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v89 /*v345*/, 0x1f0, v9
	v_and_b32_e32 v92 /*v348*/, 0x7f0, v8
	v_and_b32_e32 v93 /*v349*/, 0x9f0, v2
	v_and_b32_e32 v95 /*v351*/, 0xdf0, v6
	v_and_b32_e32 v94 /*v350*/, 0xbf0, v5
	v_and_b32_e32 v96 /*v352*/, 0xff0, v7
	v_add3_u32 v98 /*v354*/, 0x21ff0, 0, v3
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v1 /*v257*/, v250 :: v_dual_add_nc_u32 v74 /*v330*/, 0x80, v4 /*v260*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v252, v250 :: v_dual_mov_b32 v253, v250
	v_dual_mov_b32 v254, v250 :: v_dual_mov_b32 v255, v250
	v_dual_mov_b32 v242, v250 :: v_dual_mov_b32 v243, v250
	v_dual_mov_b32 v244, v250 :: v_dual_mov_b32 v245, v250
	v_dual_mov_b32 v246, v250 :: v_dual_mov_b32 v247, v250
	v_dual_mov_b32 v248, v250 :: v_dual_mov_b32 v249, v250
	v_dual_mov_b32 v234, v250 :: v_dual_mov_b32 v235, v250
	v_dual_mov_b32 v236, v250 :: v_dual_mov_b32 v237, v250
	v_dual_mov_b32 v238, v250 :: v_dual_mov_b32 v239, v250
	v_dual_mov_b32 v240, v250 :: v_dual_mov_b32 v241, v250
	v_dual_mov_b32 v226, v250 :: v_dual_mov_b32 v227, v250
	v_dual_mov_b32 v228, v250 :: v_dual_mov_b32 v229, v250
	v_dual_mov_b32 v230, v250 :: v_dual_mov_b32 v231, v250
	v_dual_mov_b32 v232, v250 :: v_dual_mov_b32 v233, v250
	v_dual_mov_b32 v218, v250 :: v_dual_mov_b32 v219, v250
	v_dual_mov_b32 v220, v250 :: v_dual_mov_b32 v221, v250
	v_dual_mov_b32 v222, v250 :: v_dual_mov_b32 v223, v250
	v_dual_mov_b32 v224, v250 :: v_dual_mov_b32 v225, v250
	v_dual_mov_b32 v210, v250 :: v_dual_mov_b32 v211, v250
	v_dual_mov_b32 v212, v250 :: v_dual_mov_b32 v213, v250
	v_dual_mov_b32 v214, v250 :: v_dual_mov_b32 v215, v250
	v_dual_mov_b32 v216, v250 :: v_dual_mov_b32 v217, v250
	v_dual_mov_b32 v202, v250 :: v_dual_mov_b32 v203, v250
	v_dual_mov_b32 v204, v250 :: v_dual_mov_b32 v205, v250
	v_dual_mov_b32 v206, v250 :: v_dual_mov_b32 v207, v250
	v_dual_mov_b32 v208, v250 :: v_dual_mov_b32 v209, v250
	v_dual_mov_b32 v194, v250 :: v_dual_mov_b32 v195, v250
	v_dual_mov_b32 v196, v250 :: v_dual_mov_b32 v197, v250
	v_dual_mov_b32 v198, v250 :: v_dual_mov_b32 v199, v250
	v_dual_mov_b32 v200, v250 :: v_dual_mov_b32 v201, v250
	v_dual_mov_b32 v186, v250 :: v_dual_mov_b32 v187, v250
	v_dual_mov_b32 v188, v250 :: v_dual_mov_b32 v189, v250
	v_dual_mov_b32 v190, v250 :: v_dual_mov_b32 v191, v250
	v_dual_mov_b32 v192, v250 :: v_dual_mov_b32 v193, v250
	v_dual_mov_b32 v178, v250 :: v_dual_mov_b32 v179, v250
	v_dual_mov_b32 v180, v250 :: v_dual_mov_b32 v181, v250
	v_dual_mov_b32 v182, v250 :: v_dual_mov_b32 v183, v250
	v_dual_mov_b32 v184, v250 :: v_dual_mov_b32 v185, v250
	v_dual_mov_b32 v170, v250 :: v_dual_mov_b32 v171, v250
	v_dual_mov_b32 v172, v250 :: v_dual_mov_b32 v173, v250
	v_dual_mov_b32 v174, v250 :: v_dual_mov_b32 v175, v250
	v_dual_mov_b32 v176, v250 :: v_dual_mov_b32 v177, v250
	v_dual_mov_b32 v162, v250 :: v_dual_mov_b32 v163, v250
	v_dual_mov_b32 v164, v250 :: v_dual_mov_b32 v165, v250
	v_dual_mov_b32 v166, v250 :: v_dual_mov_b32 v167, v250
	v_dual_mov_b32 v168, v250 :: v_dual_mov_b32 v169, v250
	v_dual_mov_b32 v154, v250 :: v_dual_mov_b32 v155, v250
	v_dual_mov_b32 v156, v250 :: v_dual_mov_b32 v157, v250
	v_dual_mov_b32 v158, v250 :: v_dual_mov_b32 v159, v250
	v_dual_mov_b32 v160, v250 :: v_dual_mov_b32 v161, v250
	v_dual_mov_b32 v146, v250 :: v_dual_mov_b32 v147, v250
	v_dual_mov_b32 v148, v250 :: v_dual_mov_b32 v149, v250
	v_dual_mov_b32 v150, v250 :: v_dual_mov_b32 v151, v250
	v_dual_mov_b32 v152, v250 :: v_dual_mov_b32 v153, v250
	v_dual_mov_b32 v138, v250 :: v_dual_mov_b32 v139, v250
	v_dual_mov_b32 v140, v250 :: v_dual_mov_b32 v141, v250
	v_dual_mov_b32 v142, v250 :: v_dual_mov_b32 v143, v250
	v_dual_mov_b32 v144, v250 :: v_dual_mov_b32 v145, v250
	v_dual_mov_b32 v130, v250 :: v_dual_mov_b32 v131, v250
	v_dual_mov_b32 v132, v250 :: v_dual_mov_b32 v133, v250
	v_dual_mov_b32 v134, v250 :: v_dual_mov_b32 v135, v250
	v_dual_mov_b32 v136, v250 :: v_dual_mov_b32 v137, v250
	v_dual_mov_b32 v122, v250 :: v_dual_mov_b32 v123, v250
	v_dual_mov_b32 v124, v250 :: v_dual_mov_b32 v125, v250
	v_dual_mov_b32 v126, v250 :: v_dual_mov_b32 v127, v250
	v_dual_mov_b32 v128, v250 :: v_dual_mov_b32 v129, v250
	v_dual_mov_b32 v114, v250 :: v_dual_mov_b32 v115, v250
	v_dual_mov_b32 v116, v250 :: v_dual_mov_b32 v117, v250
	v_dual_mov_b32 v118, v250 :: v_dual_mov_b32 v119, v250
	v_dual_mov_b32 v120, v250 :: v_dual_mov_b32 v121, v250
	v_dual_mov_b32 v106, v250 :: v_dual_mov_b32 v107, v250
	v_dual_mov_b32 v108, v250 :: v_dual_mov_b32 v109, v250
	v_dual_mov_b32 v110, v250 :: v_dual_mov_b32 v111, v250
	v_dual_mov_b32 v112, v250 :: v_dual_mov_b32 v113, v250
	v_dual_mov_b32 v98, v250 :: v_dual_mov_b32 v99, v250
	v_dual_mov_b32 v100, v250 :: v_dual_mov_b32 v101, v250
	v_dual_mov_b32 v102, v250 :: v_dual_mov_b32 v103, v250
	v_dual_mov_b32 v104, v250 :: v_dual_mov_b32 v105, v250
	v_dual_mov_b32 v90, v250 :: v_dual_mov_b32 v91, v250
	v_dual_mov_b32 v92, v250 :: v_dual_mov_b32 v93, v250
	v_dual_mov_b32 v94, v250 :: v_dual_mov_b32 v95, v250
	v_dual_mov_b32 v96, v250 :: v_dual_mov_b32 v97, v250
	v_dual_mov_b32 v82, v250 :: v_dual_mov_b32 v83, v250
	v_dual_mov_b32 v84, v250 :: v_dual_mov_b32 v85, v250
	v_dual_mov_b32 v86, v250 :: v_dual_mov_b32 v87, v250
	v_dual_mov_b32 v88, v250 :: v_dual_mov_b32 v89, v250
	v_dual_mov_b32 v74, v250 :: v_dual_mov_b32 v75, v250
	v_dual_mov_b32 v76, v250 :: v_dual_mov_b32 v77, v250
	v_dual_mov_b32 v78, v250 :: v_dual_mov_b32 v79, v250
	v_dual_mov_b32 v80, v250 :: v_dual_mov_b32 v81, v250
	v_dual_mov_b32 v66, v250 :: v_dual_mov_b32 v67, v250
	v_dual_mov_b32 v68, v250 :: v_dual_mov_b32 v69, v250
	v_dual_mov_b32 v70, v250 :: v_dual_mov_b32 v71, v250
	v_dual_mov_b32 v72, v250 :: v_dual_mov_b32 v73, v250
	v_dual_mov_b32 v58, v250 :: v_dual_mov_b32 v59, v250
	v_dual_mov_b32 v60, v250 :: v_dual_mov_b32 v61, v250
	v_dual_mov_b32 v62, v250 :: v_dual_mov_b32 v63, v250
	v_dual_mov_b32 v64, v250 :: v_dual_mov_b32 v65, v250
	v_dual_mov_b32 v50, v250 :: v_dual_mov_b32 v51, v250
	v_dual_mov_b32 v52, v250 :: v_dual_mov_b32 v53, v250
	v_dual_mov_b32 v54, v250 :: v_dual_mov_b32 v55, v250
	v_dual_mov_b32 v56, v250 :: v_dual_mov_b32 v57, v250
	v_dual_mov_b32 v42, v250 :: v_dual_mov_b32 v43, v250
	v_dual_mov_b32 v44, v250 :: v_dual_mov_b32 v45, v250
	v_dual_mov_b32 v46, v250 :: v_dual_mov_b32 v47, v250
	v_dual_mov_b32 v48, v250 :: v_dual_mov_b32 v49, v250
	v_dual_mov_b32 v34, v250 :: v_dual_mov_b32 v35, v250
	v_dual_mov_b32 v36, v250 :: v_dual_mov_b32 v37, v250
	v_dual_mov_b32 v38, v250 :: v_dual_mov_b32 v39, v250
	v_dual_mov_b32 v40, v250 :: v_dual_mov_b32 v41, v250
	v_dual_mov_b32 v26, v250 :: v_dual_mov_b32 v27, v250
	v_dual_mov_b32 v28, v250 :: v_dual_mov_b32 v29, v250
	v_dual_mov_b32 v30, v250 :: v_dual_mov_b32 v31, v250
	v_dual_mov_b32 v32, v250 :: v_dual_mov_b32 v33, v250
	v_dual_mov_b32 v18, v250 :: v_dual_mov_b32 v19, v250
	v_dual_mov_b32 v20, v250 :: v_dual_mov_b32 v21, v250
	v_dual_mov_b32 v22, v250 :: v_dual_mov_b32 v23, v250
	v_dual_mov_b32 v24, v250 :: v_dual_mov_b32 v25, v250
	v_dual_mov_b32 v10, v250 :: v_dual_mov_b32 v11, v250
	v_dual_mov_b32 v12, v250 :: v_dual_mov_b32 v13, v250
	v_dual_mov_b32 v14, v250 :: v_dual_mov_b32 v15, v250
	v_dual_mov_b32 v16, v250 :: v_dual_mov_b32 v17, v250
	v_dual_mov_b32 v2, v250 :: v_dual_mov_b32 v3, v250
	v_dual_mov_b32 v4, v250 :: v_dual_mov_b32 v5, v250
	v_dual_mov_b32 v6, v250 :: v_dual_mov_b32 v7, v250
	v_dual_mov_b32 v8, v250 :: v_dual_mov_b32 v9, v250
	s_mov_b32 s19, 1
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
.Ltmp56:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	s_max_i32 s24, s10, 0
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, s[2:3]
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, s[2:3]
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_ashrrev_i32_e32 v77 /*v333*/, 31, v76 /*v332*/
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, s[0:1]
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, s[0:1]
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	s_lshl_b32 s17, s24, 16
	.loc	1 86 51 is_stmt 0               ; f16_gemm_gfx1250.py:86:51 @[ f16_gemm_gfx1250.py:318:105 ]
	s_and_b32 s21, 1, s19
	.loc	1 92 40 is_stmt 1               ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	s_max_i32 s25, s18, 0
	.loc	1 93 16                         ; f16_gemm_gfx1250.py:93:16 @[ f16_gemm_gfx1250.py:318:105 ]
	s_add_co_i32 s19, s19, 1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_dual_mov_b32 v3 /*v259*/, v80 /*v336*/ :: v_dual_mov_b32 v5 /*v261*/, s11
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_dual_ashrrev_i32 v75 /*v331*/, 31, v74 /*v330*/ :: v_dual_mov_b32 v7 /*v263*/, v82 /*v338*/
	v_mov_b32_e32 v9 /*v265*/, s12
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, s[22:23]
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, s[16:17]
	s_lshr_b32 s24, s24, 16
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	s_lshl_b32 s26, s25, 16
	s_lshr_b32 s25, s25, 16
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, s[18:19]
	.loc	1 86 51 is_stmt 0               ; f16_gemm_gfx1250.py:86:51 @[ f16_gemm_gfx1250.py:318:105 ]
	s_cmp_eq_u32 s21, 1
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_lshl_add_u64 v[12:13] /*v[268:269]*/, v[76:77] /*v[332:333]*/, 1, s[2:3]
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, s[20:21]
	.loc	1 92 40 is_stmt 1               ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	s_mov_b32 s17, s26
	.loc	1 86 51                         ; f16_gemm_gfx1250.py:86:51 @[ f16_gemm_gfx1250.py:318:105 ]
	s_cselect_b32 s21, 0x11000, 0
	.loc	1 86 36 is_stmt 0               ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_readfirstlane_b32 s44, v2 /*v258*/
	v_readfirstlane_b32 s27, v3 /*v259*/
	v_readfirstlane_b32 s29, v5 /*v261*/
	.loc	1 92 40 is_stmt 1               ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_readfirstlane_b32 s48, v6 /*v262*/
	v_readfirstlane_b32 s39, v7 /*v263*/
	v_readfirstlane_b32 s41, v9 /*v265*/
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, s[22:23]
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_or_b32_e32 v20 /*v276*/, s24, v83 /*v339*/
.Ltmp57:
	.loc	1 102 23                        ; f16_gemm_gfx1250.py:102:23 @[ f16_gemm_gfx1250.py:320:92 ]
	s_cselect_b32 s24, 0, 0x11000
.Ltmp58:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, s[16:17]
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, s[20:21]
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, s[18:19]
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_dual_add_nc_u32 v5 /*v261*/, s21, v79 /*v335*/ :: v_dual_bitop2_b32 v15 /*v271*/, s25, v84 /*v340*/ bitop3:0x54
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_lshl_add_u64 v[18:19] /*v[274:275]*/, v[74:75] /*v[330:331]*/, 1, s[4:5]
	v_add_nc_u32_e32 v7 /*v263*/, s21, v81 /*v337*/
.Ltmp59:
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	v_dual_add_nc_u32 v107 /*v363*/, s24, v97 /*v353*/ :: v_dual_add_nc_u32 v21 /*v277*/, s24, v98 /*v354*/
.Ltmp60:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_readfirstlane_b32 s25, v11 /*v267*/
	v_or_b32_e32 v11 /*v267*/, 0x80000000, v13 /*v269*/
	v_mov_b32_e32 v4 /*v260*/, v20 /*v276*/
	v_readfirstlane_b32 s28, v14 /*v270*/
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b32_e32 v14 /*v270*/, v18 /*v274*/
	v_readfirstlane_b32 s37, v3 /*v259*/
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_dual_mov_b32 v3 /*v259*/, v5 /*v261*/ :: v_dual_mov_b32 v5 /*v261*/, v7 /*v263*/
.Ltmp61:
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	v_dual_add_nc_u32 v77 /*v333*/, v21 /*v277*/, v89 /*v345*/ :: v_dual_add_nc_u32 v106 /*v362*/, v21 /*v277*/, v96 /*v352*/
.Ltmp62:
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b32_e32 v7 /*v263*/, v11 /*v267*/
	v_readfirstlane_b32 s24, v10 /*v266*/
	v_readfirstlane_b32 s30, v16 /*v272*/
	v_readfirstlane_b32 s31, v17 /*v273*/
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_mov_b32_e32 v10 /*v266*/, v15 /*v271*/
	.loc	1 86 36                         ; f16_gemm_gfx1250.py:86:36 @[ f16_gemm_gfx1250.py:318:105 ]
	v_readfirstlane_b32 s26, v4 /*v260*/
	v_readfirstlane_b32 s46, v12 /*v268*/
	v_readfirstlane_b32 s45, v3 /*v259*/
	v_readfirstlane_b32 s47, v7 /*v263*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[44:47], s[24:31]
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	v_or_b32_e32 v13 /*v269*/, 0x80000000, v19 /*v275*/
	v_readfirstlane_b32 s43, v9 /*v265*/
	v_readfirstlane_b32 s36, v2 /*v258*/
	v_readfirstlane_b32 s40, v6 /*v262*/
	v_readfirstlane_b32 s42, v8 /*v264*/
	v_dual_mov_b32 v9 /*v265*/, v13 /*v269*/ :: v_dual_add_nc_u32 v116 /*v372*/, v107 /*v363*/, v87 /*v343*/
	v_readfirstlane_b32 s38, v10 /*v266*/
	v_readfirstlane_b32 s50, v14 /*v270*/
	v_readfirstlane_b32 s49, v5 /*v261*/
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s51, v9 /*v265*/
.Ltmp63:
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	v_dual_add_nc_u32 v99 /*v355*/, v107 /*v363*/, v85 /*v341*/ :: v_dual_add_nc_u32 v75 /*v331*/, v107 /*v363*/, v86 /*v342*/
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	v_dual_add_nc_u32 v100 /*v356*/, v21 /*v277*/, v90 /*v346*/ :: v_dual_add_nc_u32 v101 /*v357*/, v21 /*v277*/, v91 /*v347*/
	v_dual_add_nc_u32 v102 /*v358*/, v21 /*v277*/, v92 /*v348*/ :: v_dual_add_nc_u32 v103 /*v359*/, v21 /*v277*/, v93 /*v349*/
	v_dual_add_nc_u32 v104 /*v360*/, v21 /*v277*/, v94 /*v350*/ :: v_dual_add_nc_u32 v105 /*v361*/, v21 /*v277*/, v95 /*v351*/
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	v_add_nc_u32_e32 v117 /*v373*/, v107 /*v363*/, v88 /*v344*/
.Ltmp64:
	.loc	1 317 22                        ; f16_gemm_gfx1250.py:317:22
	v_add_nc_u32_e32 v74 /*v330*/, 0x80, v74 /*v330*/
	v_add_nc_u32_e32 v76 /*v332*/, 0x80, v76 /*v332*/
	s_addk_co_i32 s18, 0xff80
	s_addk_co_i32 s10, 0xff80
	s_cmp_lg_u32 s15, s19
.Ltmp65:
	.loc	1 92 40                         ; f16_gemm_gfx1250.py:92:40 @[ f16_gemm_gfx1250.py:318:105 ]
	tensor_load_to_lds s[48:51], s[36:43]
.Ltmp66:
	.loc	1 100 36                        ; f16_gemm_gfx1250.py:100:36 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_tensorcnt 0x2
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[66:69] /*v[322:325]*/, v99 /*v355*/
	ds_load_b128 v[70:73] /*v[326:329]*/, v99 /*v355*/ offset:32
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[10:13] /*v[266:269]*/, v77 /*v333*/
	ds_load_b128 v[14:17] /*v[270:273]*/, v77 /*v333*/ offset:32
	ds_load_b128 v[2:5] /*v[258:261]*/, v100 /*v356*/ offset:8192
	ds_load_b128 v[6:9] /*v[262:265]*/, v100 /*v356*/ offset:8224
	ds_load_b128 v[18:21] /*v[274:277]*/, v101 /*v357*/ offset:16384
	ds_load_b128 v[22:25] /*v[278:281]*/, v101 /*v357*/ offset:16416
	ds_load_b128 v[26:29] /*v[282:285]*/, v102 /*v358*/ offset:24576
	ds_load_b128 v[30:33] /*v[286:289]*/, v102 /*v358*/ offset:24608
	ds_load_b128 v[34:37] /*v[290:293]*/, v103 /*v359*/ offset:32768
	ds_load_b128 v[38:41] /*v[294:297]*/, v103 /*v359*/ offset:32800
	ds_load_b128 v[42:45] /*v[298:301]*/, v104 /*v360*/ offset:40960
	ds_load_b128 v[46:49] /*v[302:305]*/, v104 /*v360*/ offset:40992
	ds_load_b128 v[58:61] /*v[314:317]*/, v105 /*v361*/ offset:49152
	ds_load_b128 v[62:65] /*v[318:321]*/, v105 /*v361*/ offset:49184
	ds_load_b128 v[50:53] /*v[306:309]*/, v106 /*v362*/ offset:57344
	ds_load_b128 v[54:57] /*v[310:313]*/, v106 /*v362*/ offset:57376
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v75 /*v331*/ offset:16384
	ds_load_b128 v[112:115] /*v[368:371]*/, v75 /*v331*/ offset:16416
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x10
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[10:17] /*v[266:273]*/, v[66:73] /*v[322:329]*/, v[250:257]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_f16 v[242:249], v[2:9] /*v[258:265]*/, v[66:73] /*v[322:329]*/, v[242:249]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_f16 v[234:241], v[18:25] /*v[274:281]*/, v[66:73] /*v[322:329]*/, v[234:241]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_f16 v[226:233], v[26:33] /*v[282:289]*/, v[66:73] /*v[322:329]*/, v[226:233]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_f16 v[218:225], v[34:41] /*v[290:297]*/, v[66:73] /*v[322:329]*/, v[218:225]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_f16 v[210:217], v[42:49] /*v[298:305]*/, v[66:73] /*v[322:329]*/, v[210:217]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_f16 v[202:209], v[58:65] /*v[314:321]*/, v[66:73] /*v[322:329]*/, v[202:209]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_f16 v[194:201], v[50:57] /*v[306:313]*/, v[66:73] /*v[322:329]*/, v[194:201]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[66:69] /*v[322:325]*/, v116 /*v372*/ offset:32768
	ds_load_b128 v[70:73] /*v[326:329]*/, v116 /*v372*/ offset:32800
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[10:17] /*v[266:273]*/, v[108:115] /*v[364:371]*/, v[186:193]
	v_wmma_f32_16x16x32_f16 v[178:185], v[2:9] /*v[258:265]*/, v[108:115] /*v[364:371]*/, v[178:185]
	v_wmma_f32_16x16x32_f16 v[170:177], v[18:25] /*v[274:281]*/, v[108:115] /*v[364:371]*/, v[170:177]
	v_wmma_f32_16x16x32_f16 v[162:169], v[26:33] /*v[282:289]*/, v[108:115] /*v[364:371]*/, v[162:169]
	v_wmma_f32_16x16x32_f16 v[154:161], v[34:41] /*v[290:297]*/, v[108:115] /*v[364:371]*/, v[154:161]
	v_wmma_f32_16x16x32_f16 v[146:153], v[42:49] /*v[298:305]*/, v[108:115] /*v[364:371]*/, v[146:153]
	v_wmma_f32_16x16x32_f16 v[138:145], v[58:65] /*v[314:321]*/, v[108:115] /*v[364:371]*/, v[138:145]
	v_wmma_f32_16x16x32_f16 v[130:137], v[50:57] /*v[306:313]*/, v[108:115] /*v[364:371]*/, v[130:137]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v117 /*v373*/ offset:49152
	ds_load_b128 v[112:115] /*v[368:371]*/, v117 /*v373*/ offset:49184
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[10:17] /*v[266:273]*/, v[66:73] /*v[322:329]*/, v[122:129]
	v_wmma_f32_16x16x32_f16 v[114:121], v[2:9] /*v[258:265]*/, v[66:73] /*v[322:329]*/, v[114:121]
	v_wmma_f32_16x16x32_f16 v[106:113], v[18:25] /*v[274:281]*/, v[66:73] /*v[322:329]*/, v[106:113]
	v_wmma_f32_16x16x32_f16 v[98:105], v[26:33] /*v[282:289]*/, v[66:73] /*v[322:329]*/, v[98:105]
	v_wmma_f32_16x16x32_f16 v[90:97], v[34:41] /*v[290:297]*/, v[66:73] /*v[322:329]*/, v[90:97]
	v_wmma_f32_16x16x32_f16 v[82:89], v[42:49] /*v[298:305]*/, v[66:73] /*v[322:329]*/, v[82:89]
	v_wmma_f32_16x16x32_f16 v[74:81], v[58:65] /*v[314:321]*/, v[66:73] /*v[322:329]*/, v[74:81]
	v_wmma_f32_16x16x32_f16 v[66:73], v[50:57] /*v[306:313]*/, v[66:73] /*v[322:329]*/, v[66:73]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_f16 v[58:65], v[10:17] /*v[266:273]*/, v[108:115] /*v[364:371]*/, v[58:65]
	v_wmma_f32_16x16x32_f16 v[50:57], v[2:9] /*v[258:265]*/, v[108:115] /*v[364:371]*/, v[50:57]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[2:5] /*v[258:261]*/, v99 /*v355*/ offset:64
	ds_load_b128 v[6:9] /*v[262:265]*/, v99 /*v355*/ offset:96
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[10:13] /*v[266:269]*/, v77 /*v333*/ offset:64
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[42:49], v[18:25] /*v[274:281]*/, v[108:115] /*v[364:371]*/, v[42:49]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[14:17] /*v[270:273]*/, v77 /*v333*/ offset:96
	ds_load_b128 v[18:21] /*v[274:277]*/, v100 /*v356*/ offset:8256
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[34:41], v[26:33] /*v[282:289]*/, v[108:115] /*v[364:371]*/, v[34:41]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[22:25] /*v[278:281]*/, v100 /*v356*/ offset:8288
	ds_load_b128 v[26:29] /*v[282:285]*/, v101 /*v357*/ offset:16448
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[26:33], v[34:41] /*v[290:297]*/, v[108:115] /*v[364:371]*/, v[26:33]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[30:33] /*v[286:289]*/, v101 /*v357*/ offset:16480
	ds_load_b128 v[34:37] /*v[290:293]*/, v102 /*v358*/ offset:24640
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[18:25], v[42:49] /*v[298:305]*/, v[108:115] /*v[364:371]*/, v[18:25]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[38:41] /*v[294:297]*/, v102 /*v358*/ offset:24672
	ds_load_b128 v[42:45] /*v[298:301]*/, v103 /*v359*/ offset:32832
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[10:17], v[58:65] /*v[314:321]*/, v[108:115] /*v[364:371]*/, v[10:17]
	v_wmma_f32_16x16x32_f16 v[2:9], v[50:57] /*v[306:313]*/, v[108:115] /*v[364:371]*/, v[2:9]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v103 /*v359*/ offset:32864
	ds_load_b128 v[50:53] /*v[306:309]*/, v104 /*v360*/ offset:41024
	ds_load_b128 v[54:57] /*v[310:313]*/, v104 /*v360*/ offset:41056
	ds_load_b128 v[58:61] /*v[314:317]*/, v105 /*v361*/ offset:49216
	ds_load_b128 v[62:65] /*v[318:321]*/, v105 /*v361*/ offset:49248
	ds_load_b128 v[66:69] /*v[322:325]*/, v106 /*v362*/ offset:57408
	ds_load_b128 v[70:73] /*v[326:329]*/, v106 /*v362*/ offset:57440
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v75 /*v331*/ offset:16448
	ds_load_b128 v[112:115] /*v[368:371]*/, v75 /*v331*/ offset:16480
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x10
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[250:257]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_f16 v[242:249], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[242:249]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_f16 v[234:241], v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[234:241]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_f16 v[226:233], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[226:233]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_f16 v[218:225], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[218:225]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_f16 v[210:217], v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/, v[210:217]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_f16 v[202:209], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[202:209]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_f16 v[194:201], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[194:201]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[2:5] /*v[258:261]*/, v116 /*v372*/ offset:32832
	ds_load_b128 v[6:9] /*v[262:265]*/, v116 /*v372*/ offset:32864
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[10:17] /*v[266:273]*/, v[108:115] /*v[364:371]*/, v[186:193]
	v_wmma_f32_16x16x32_f16 v[178:185], v[18:25] /*v[274:281]*/, v[108:115] /*v[364:371]*/, v[178:185]
	v_wmma_f32_16x16x32_f16 v[170:177], v[26:33] /*v[282:289]*/, v[108:115] /*v[364:371]*/, v[170:177]
	v_wmma_f32_16x16x32_f16 v[162:169], v[34:41] /*v[290:297]*/, v[108:115] /*v[364:371]*/, v[162:169]
	v_wmma_f32_16x16x32_f16 v[154:161], v[42:49] /*v[298:305]*/, v[108:115] /*v[364:371]*/, v[154:161]
	v_wmma_f32_16x16x32_f16 v[146:153], v[50:57] /*v[306:313]*/, v[108:115] /*v[364:371]*/, v[146:153]
	v_wmma_f32_16x16x32_f16 v[138:145], v[58:65] /*v[314:321]*/, v[108:115] /*v[364:371]*/, v[138:145]
	v_wmma_f32_16x16x32_f16 v[130:137], v[66:73] /*v[322:329]*/, v[108:115] /*v[364:371]*/, v[130:137]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v117 /*v373*/ offset:49216
	ds_load_b128 v[112:115] /*v[368:371]*/, v117 /*v373*/ offset:49248
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[122:129]
	v_wmma_f32_16x16x32_f16 v[114:121], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[114:121]
	v_wmma_f32_16x16x32_f16 v[106:113], v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[106:113]
	v_wmma_f32_16x16x32_f16 v[98:105], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[98:105]
	v_wmma_f32_16x16x32_f16 v[90:97], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[90:97]
	v_wmma_f32_16x16x32_f16 v[82:89], v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/, v[82:89]
	v_wmma_f32_16x16x32_f16 v[74:81], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[74:81]
	v_wmma_f32_16x16x32_f16 v[66:73], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[66:73]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_f16 v[58:65], v[10:17] /*v[266:273]*/, v[108:115] /*v[364:371]*/, v[58:65]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[2:5] /*v[258:261]*/, v99 /*v355*/ offset:128
	ds_load_b128 v[6:9] /*v[262:265]*/, v99 /*v355*/ offset:160
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[10:13] /*v[266:269]*/, v77 /*v333*/ offset:128
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[50:57], v[18:25] /*v[274:281]*/, v[108:115] /*v[364:371]*/, v[50:57]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[14:17] /*v[270:273]*/, v77 /*v333*/ offset:160
	ds_load_b128 v[18:21] /*v[274:277]*/, v100 /*v356*/ offset:8320
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[42:49], v[26:33] /*v[282:289]*/, v[108:115] /*v[364:371]*/, v[42:49]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[22:25] /*v[278:281]*/, v100 /*v356*/ offset:8352
	ds_load_b128 v[26:29] /*v[282:285]*/, v101 /*v357*/ offset:16512
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[34:41], v[34:41] /*v[290:297]*/, v[108:115] /*v[364:371]*/, v[34:41]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[30:33] /*v[286:289]*/, v101 /*v357*/ offset:16544
	ds_load_b128 v[34:37] /*v[290:293]*/, v102 /*v358*/ offset:24704
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[26:33], v[42:49] /*v[298:305]*/, v[108:115] /*v[364:371]*/, v[26:33]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[38:41] /*v[294:297]*/, v102 /*v358*/ offset:24736
	ds_load_b128 v[42:45] /*v[298:301]*/, v103 /*v359*/ offset:32896
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[18:25], v[50:57] /*v[306:313]*/, v[108:115] /*v[364:371]*/, v[18:25]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v103 /*v359*/ offset:32928
	ds_load_b128 v[50:53] /*v[306:309]*/, v104 /*v360*/ offset:41088
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[10:17], v[58:65] /*v[314:321]*/, v[108:115] /*v[364:371]*/, v[10:17]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[54:57] /*v[310:313]*/, v104 /*v360*/ offset:41120
	ds_load_b128 v[58:61] /*v[314:317]*/, v105 /*v361*/ offset:49280
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[2:9], v[66:73] /*v[322:329]*/, v[108:115] /*v[364:371]*/, v[2:9]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[62:65] /*v[318:321]*/, v105 /*v361*/ offset:49312
	ds_load_b128 v[66:69] /*v[322:325]*/, v106 /*v362*/ offset:57472
	ds_load_b128 v[70:73] /*v[326:329]*/, v106 /*v362*/ offset:57504
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v75 /*v331*/ offset:16512
	ds_load_b128 v[112:115] /*v[368:371]*/, v75 /*v331*/ offset:16544
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x10
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[250:257]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_f16 v[242:249], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[242:249]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_f16 v[234:241], v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[234:241]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_f16 v[226:233], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[226:233]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_f16 v[218:225], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[218:225]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_f16 v[210:217], v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/, v[210:217]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_f16 v[202:209], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[202:209]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_f16 v[194:201], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[194:201]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[2:5] /*v[258:261]*/, v116 /*v372*/ offset:32896
	ds_load_b128 v[6:9] /*v[262:265]*/, v116 /*v372*/ offset:32928
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[10:17] /*v[266:273]*/, v[108:115] /*v[364:371]*/, v[186:193]
	v_wmma_f32_16x16x32_f16 v[178:185], v[18:25] /*v[274:281]*/, v[108:115] /*v[364:371]*/, v[178:185]
	v_wmma_f32_16x16x32_f16 v[170:177], v[26:33] /*v[282:289]*/, v[108:115] /*v[364:371]*/, v[170:177]
	v_wmma_f32_16x16x32_f16 v[162:169], v[34:41] /*v[290:297]*/, v[108:115] /*v[364:371]*/, v[162:169]
	v_wmma_f32_16x16x32_f16 v[154:161], v[42:49] /*v[298:305]*/, v[108:115] /*v[364:371]*/, v[154:161]
	v_wmma_f32_16x16x32_f16 v[146:153], v[50:57] /*v[306:313]*/, v[108:115] /*v[364:371]*/, v[146:153]
	v_wmma_f32_16x16x32_f16 v[138:145], v[58:65] /*v[314:321]*/, v[108:115] /*v[364:371]*/, v[138:145]
	v_wmma_f32_16x16x32_f16 v[130:137], v[66:73] /*v[322:329]*/, v[108:115] /*v[364:371]*/, v[130:137]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v117 /*v373*/ offset:49280
	ds_load_b128 v[112:115] /*v[368:371]*/, v117 /*v373*/ offset:49312
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[122:129]
	v_wmma_f32_16x16x32_f16 v[114:121], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[114:121]
	v_wmma_f32_16x16x32_f16 v[106:113], v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[106:113]
	v_wmma_f32_16x16x32_f16 v[98:105], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[98:105]
	v_wmma_f32_16x16x32_f16 v[90:97], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[90:97]
	v_wmma_f32_16x16x32_f16 v[82:89], v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/, v[82:89]
	v_wmma_f32_16x16x32_f16 v[74:81], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[74:81]
	v_wmma_f32_16x16x32_f16 v[66:73], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[66:73]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_f16 v[58:65], v[10:17] /*v[266:273]*/, v[108:115] /*v[364:371]*/, v[58:65]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[2:5] /*v[258:261]*/, v99 /*v355*/ offset:192
	ds_load_b128 v[6:9] /*v[262:265]*/, v99 /*v355*/ offset:224
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[10:13] /*v[266:269]*/, v77 /*v333*/ offset:192
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[50:57], v[18:25] /*v[274:281]*/, v[108:115] /*v[364:371]*/, v[50:57]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[14:17] /*v[270:273]*/, v77 /*v333*/ offset:224
	ds_load_b128 v[18:21] /*v[274:277]*/, v100 /*v356*/ offset:8384
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[42:49], v[26:33] /*v[282:289]*/, v[108:115] /*v[364:371]*/, v[42:49]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[22:25] /*v[278:281]*/, v100 /*v356*/ offset:8416
	ds_load_b128 v[26:29] /*v[282:285]*/, v101 /*v357*/ offset:16576
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[34:41], v[34:41] /*v[290:297]*/, v[108:115] /*v[364:371]*/, v[34:41]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[30:33] /*v[286:289]*/, v101 /*v357*/ offset:16608
	ds_load_b128 v[34:37] /*v[290:293]*/, v102 /*v358*/ offset:24768
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[26:33], v[42:49] /*v[298:305]*/, v[108:115] /*v[364:371]*/, v[26:33]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[38:41] /*v[294:297]*/, v102 /*v358*/ offset:24800
	ds_load_b128 v[42:45] /*v[298:301]*/, v103 /*v359*/ offset:32960
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[18:25], v[50:57] /*v[306:313]*/, v[108:115] /*v[364:371]*/, v[18:25]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v103 /*v359*/ offset:32992
	ds_load_b128 v[50:53] /*v[306:309]*/, v104 /*v360*/ offset:41152
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[10:17], v[58:65] /*v[314:321]*/, v[108:115] /*v[364:371]*/, v[10:17]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[54:57] /*v[310:313]*/, v104 /*v360*/ offset:41184
	ds_load_b128 v[58:61] /*v[314:317]*/, v105 /*v361*/ offset:49344
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	v_wmma_f32_16x16x32_f16 v[2:9], v[66:73] /*v[322:329]*/, v[108:115] /*v[364:371]*/, v[2:9]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[62:65] /*v[318:321]*/, v105 /*v361*/ offset:49376
	ds_load_b128 v[66:69] /*v[322:325]*/, v106 /*v362*/ offset:57536
	ds_load_b128 v[70:73] /*v[326:329]*/, v106 /*v362*/ offset:57568
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[100:103] /*v[356:359]*/, v75 /*v331*/ offset:16576
	ds_load_b128 v[104:107] /*v[360:363]*/, v75 /*v331*/ offset:16608
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x10
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[250:257]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_f16 v[242:249], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[242:249]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_f16 v[234:241], v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[234:241]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_f16 v[226:233], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[226:233]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_f16 v[218:225], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[218:225]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_f16 v[210:217], v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/, v[210:217]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_f16 v[202:209], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[202:209]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_f16 v[194:201], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[194:201]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[2:5] /*v[258:261]*/, v116 /*v372*/ offset:32960
	ds_load_b128 v[6:9] /*v[262:265]*/, v116 /*v372*/ offset:32992
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[10:17] /*v[266:273]*/, v[100:107] /*v[356:363]*/, v[186:193]
	v_wmma_f32_16x16x32_f16 v[178:185], v[18:25] /*v[274:281]*/, v[100:107] /*v[356:363]*/, v[178:185]
	v_wmma_f32_16x16x32_f16 v[170:177], v[26:33] /*v[282:289]*/, v[100:107] /*v[356:363]*/, v[170:177]
	v_wmma_f32_16x16x32_f16 v[162:169], v[34:41] /*v[290:297]*/, v[100:107] /*v[356:363]*/, v[162:169]
	v_wmma_f32_16x16x32_f16 v[154:161], v[42:49] /*v[298:305]*/, v[100:107] /*v[356:363]*/, v[154:161]
	v_wmma_f32_16x16x32_f16 v[146:153], v[50:57] /*v[306:313]*/, v[100:107] /*v[356:363]*/, v[146:153]
	v_wmma_f32_16x16x32_f16 v[138:145], v[58:65] /*v[314:321]*/, v[100:107] /*v[356:363]*/, v[138:145]
	v_wmma_f32_16x16x32_f16 v[130:137], v[66:73] /*v[322:329]*/, v[100:107] /*v[356:363]*/, v[130:137]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:320:92 ]
	ds_load_b128 v[100:103] /*v[356:359]*/, v117 /*v373*/ offset:49344
	ds_load_b128 v[104:107] /*v[360:363]*/, v117 /*v373*/ offset:49376
	.loc	1 108 46                        ; f16_gemm_gfx1250.py:108:46 @[ f16_gemm_gfx1250.py:320:92 ]
	s_wait_dscnt 0x2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[122:129]
	v_wmma_f32_16x16x32_f16 v[114:121], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[114:121]
	v_wmma_f32_16x16x32_f16 v[106:113], v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[106:113]
	v_wmma_f32_16x16x32_f16 v[98:105], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[98:105]
	v_wmma_f32_16x16x32_f16 v[90:97], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[90:97]
	v_wmma_f32_16x16x32_f16 v[82:89], v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/, v[82:89]
	v_wmma_f32_16x16x32_f16 v[74:81], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[74:81]
	v_wmma_f32_16x16x32_f16 v[66:73], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[66:73]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_f16 v[58:65], v[10:17] /*v[266:273]*/, v[100:107] /*v[356:363]*/, v[58:65]
	v_wmma_f32_16x16x32_f16 v[50:57], v[18:25] /*v[274:281]*/, v[100:107] /*v[356:363]*/, v[50:57]
	v_wmma_f32_16x16x32_f16 v[42:49], v[26:33] /*v[282:289]*/, v[100:107] /*v[356:363]*/, v[42:49]
	v_wmma_f32_16x16x32_f16 v[34:41], v[34:41] /*v[290:297]*/, v[100:107] /*v[356:363]*/, v[34:41]
	v_wmma_f32_16x16x32_f16 v[26:33], v[42:49] /*v[298:305]*/, v[100:107] /*v[356:363]*/, v[26:33]
	v_wmma_f32_16x16x32_f16 v[18:25], v[50:57] /*v[306:313]*/, v[100:107] /*v[356:363]*/, v[18:25]
	v_wmma_f32_16x16x32_f16 v[10:17], v[58:65] /*v[314:321]*/, v[100:107] /*v[356:363]*/, v[10:17]
	v_wmma_f32_16x16x32_f16 v[2:9], v[66:73] /*v[322:329]*/, v[100:107] /*v[356:363]*/, v[2:9]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp67:
	.loc	1 317 22                        ; f16_gemm_gfx1250.py:317:22
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %._crit_edge.loopexit
	.loc	1 0 22 is_stmt 0                ; f16_gemm_gfx1250.py:0:22
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b32_e32 v3 /*v259*/, v78 /*v334*/
.Ltmp68:
	.loc	1 102 23 is_stmt 1              ; f16_gemm_gfx1250.py:102:23 @[ f16_gemm_gfx1250.py:324:96 ]
	s_bitcmp1_b32 s15, 0
	s_cselect_b32 s22, 0, 0x8800
.LBB0_7:                                ; %._crit_edge
	.loc	1 0 23 is_stmt 0                ; f16_gemm_gfx1250.py:0:23
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 102 52 is_stmt 1              ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_and_b32_e32 v250 /*v506*/, 15, v0
	v_and_b32_e32 v251 /*v507*/, 0xc0, v0
	.loc	1 102 23 is_stmt 0              ; f16_gemm_gfx1250.py:102:23 @[ f16_gemm_gfx1250.py:324:96 ]
	s_lshl_b32 s0, s22, 1
	.loc	1 100 36 is_stmt 1              ; f16_gemm_gfx1250.py:100:36 @[ f16_gemm_gfx1250.py:324:96 ]
	s_wait_tensorcnt 0x0
	.loc	1 102 23                        ; f16_gemm_gfx1250.py:102:23 @[ f16_gemm_gfx1250.py:324:96 ]
	s_add_co_i32 s0, s0, 0
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	.loc	1 102 52 is_stmt 0              ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_lshlrev_b32_e32 v2 /*v258*/, 8, v250 /*v506*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x51                     ;  msbs: dst=1 src0=1 src1=0 src2=1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v4 /*v260*/, v251 /*v507*/, 6, v2 /*v258*/
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
.Ltmp69:
	.loc	1 326 47 is_stmt 1              ; f16_gemm_gfx1250.py:326:47
	v_lshrrev_b32_e32 v251 /*v507*/, 2, v251 /*v507*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 327 47                        ; f16_gemm_gfx1250.py:327:47
	v_lshrrev_b32_e32 v0, 1, v0
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp70:
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_or_b32_e32 v3 /*v259*/, v4 /*v260*/, v3 /*v259*/
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_or_b32_e32 v1, v1, v2 /*v258*/
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_lshrrev_b32_e32 v4 /*v260*/, 4, v4 /*v260*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp71:
	.loc	1 327 32                        ; f16_gemm_gfx1250.py:327:32
	v_and_or_b32 v0, v0, 24, s14
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
.Ltmp72:
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_or_b32_e32 v5 /*v261*/, 0x4000, v3 /*v259*/
	v_add_nc_u32_e32 v6 /*v262*/, s0, v3 /*v259*/
	v_or_b32_e32 v7 /*v263*/, 0x8000, v3 /*v259*/
	v_or_b32_e32 v3 /*v259*/, 0xc000, v3 /*v259*/
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_lshrrev_b32_e32 v10 /*v266*/, 4, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_dual_lshrrev_b32 v5 /*v261*/, 4, v5 /*v261*/ :: v_dual_add_nc_u32 v4 /*v260*/, v6 /*v262*/, v4 /*v260*/
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_or_b32_e32 v11 /*v267*/, 0x2000, v1
	v_add3_u32 v252 /*v508*/, 0x21ff0, s0, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_and_b32_e32 v10 /*v266*/, 0x1f0, v10 /*v266*/
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_lshrrev_b32_e32 v3 /*v259*/, 4, v3 /*v259*/
	v_and_b32_e32 v5 /*v261*/, 0x7fffff0, v5 /*v261*/
	v_lshrrev_b32_e32 v7 /*v263*/, 4, v7 /*v263*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_dual_lshrrev_b32 v35 /*v291*/, 4, v11 /*v267*/ :: v_dual_add_nc_u32 v34 /*v290*/, v252 /*v508*/, v10 /*v266*/
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	v_and_b32_e32 v3 /*v259*/, 0x7fffff0, v3 /*v259*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v5 /*v261*/, v6 /*v262*/, v5 /*v261*/
	v_and_b32_e32 v7 /*v263*/, 0x7fffff0, v7 /*v263*/
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_or_b32_e32 v36 /*v292*/, 0x4000, v1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_and_b32_e32 v35 /*v291*/, 0x3f0, v35 /*v291*/
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_or_b32_e32 v172 /*v428*/, 0x6000, v1
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	ds_load_b128 v[162:165] /*v[418:421]*/, v4 /*v260*/
	ds_load_b128 v[166:169] /*v[422:425]*/, v4 /*v260*/ offset:32
	ds_load_b128 v[154:157] /*v[410:413]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[158:161] /*v[414:417]*/, v4 /*v260*/ offset:96
	ds_load_b128 v[146:149] /*v[402:405]*/, v4 /*v260*/ offset:128
	ds_load_b128 v[150:153] /*v[406:409]*/, v4 /*v260*/ offset:160
	ds_load_b128 v[138:141] /*v[394:397]*/, v4 /*v260*/ offset:192
	ds_load_b128 v[142:145] /*v[398:401]*/, v4 /*v260*/ offset:224
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_add_nc_u32 v4 /*v260*/, v6 /*v262*/, v7 /*v263*/ :: v_dual_lshrrev_b32 v36 /*v292*/, 4, v36 /*v292*/
	v_dual_add_nc_u32 v14 /*v270*/, v6 /*v262*/, v3 /*v259*/ :: v_dual_lshrrev_b32 v172 /*v428*/, 4, v172 /*v428*/
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_add_nc_u32_e32 v170 /*v426*/, v252 /*v508*/, v35 /*v291*/
	.loc	1 102 52                        ; f16_gemm_gfx1250.py:102:52 @[ f16_gemm_gfx1250.py:324:96 ]
	ds_load_b128 v[130:133] /*v[386:389]*/, v5 /*v261*/ offset:16384
	ds_load_b128 v[134:137] /*v[390:393]*/, v5 /*v261*/ offset:16416
	ds_load_b128 v[122:125] /*v[378:381]*/, v5 /*v261*/ offset:16448
	ds_load_b128 v[126:129] /*v[382:385]*/, v5 /*v261*/ offset:16480
	ds_load_b128 v[114:117] /*v[370:373]*/, v5 /*v261*/ offset:16512
	ds_load_b128 v[118:121] /*v[374:377]*/, v5 /*v261*/ offset:16544
	ds_load_b128 v[106:109] /*v[362:365]*/, v5 /*v261*/ offset:16576
	ds_load_b128 v[110:113] /*v[366:369]*/, v5 /*v261*/ offset:16608
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:32768
	ds_load_b128 v[86:89] /*v[342:345]*/, v4 /*v260*/ offset:32800
	ds_load_b128 v[98:101] /*v[354:357]*/, v4 /*v260*/ offset:32832
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/ offset:32864
	ds_load_b128 v[90:93] /*v[346:349]*/, v4 /*v260*/ offset:32896
	ds_load_b128 v[94:97] /*v[350:353]*/, v4 /*v260*/ offset:32928
	ds_load_b128 v[74:77] /*v[330:333]*/, v4 /*v260*/ offset:32960
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/ offset:32992
	ds_load_b128 v[2:5] /*v[258:261]*/, v14 /*v270*/ offset:49152
	ds_load_b128 v[6:9] /*v[262:265]*/, v14 /*v270*/ offset:49184
	ds_load_b128 v[26:29] /*v[282:285]*/, v14 /*v270*/ offset:49216
	ds_load_b128 v[30:33] /*v[286:289]*/, v14 /*v270*/ offset:49248
	ds_load_b128 v[18:21] /*v[274:277]*/, v14 /*v270*/ offset:49280
	ds_load_b128 v[22:25] /*v[278:281]*/, v14 /*v270*/ offset:49312
	ds_load_b128 v[10:13] /*v[266:269]*/, v14 /*v270*/ offset:49344
	ds_load_b128 v[14:17] /*v[270:273]*/, v14 /*v270*/ offset:49376
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	ds_load_b128 v[58:61] /*v[314:317]*/, v34 /*v290*/
	ds_load_b128 v[62:65] /*v[318:321]*/, v34 /*v290*/ offset:32
	ds_load_b128 v[66:69] /*v[322:325]*/, v34 /*v290*/ offset:64
	ds_load_b128 v[70:73] /*v[326:329]*/, v34 /*v290*/ offset:96
	ds_load_b128 v[50:53] /*v[306:309]*/, v34 /*v290*/ offset:128
	ds_load_b128 v[54:57] /*v[310:313]*/, v34 /*v290*/ offset:160
	ds_load_b128 v[42:45] /*v[298:301]*/, v34 /*v290*/ offset:192
	ds_load_b128 v[46:49] /*v[302:305]*/, v34 /*v290*/ offset:224
	v_and_b32_e32 v171 /*v427*/, 0x5f0, v36 /*v292*/
	ds_load_b128 v[34:37] /*v[290:293]*/, v170 /*v426*/ offset:8192
	ds_load_b128 v[38:41] /*v[294:297]*/, v170 /*v426*/ offset:8224
	ds_load_b128 v[242:245] /*v[498:501]*/, v170 /*v426*/ offset:8256
	ds_load_b128 v[246:249] /*v[502:505]*/, v170 /*v426*/ offset:8288
	ds_load_b128 v[234:237] /*v[490:493]*/, v170 /*v426*/ offset:8320
	ds_load_b128 v[238:241] /*v[494:497]*/, v170 /*v426*/ offset:8352
	ds_load_b128 v[226:229] /*v[482:485]*/, v170 /*v426*/ offset:8384
	ds_load_b128 v[230:233] /*v[486:489]*/, v170 /*v426*/ offset:8416
	v_and_b32_e32 v170 /*v426*/, 0x7f0, v172 /*v428*/
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_or_b32_e32 v172 /*v428*/, 0x8000, v1
	v_or_b32_e32 v255 /*v511*/, 0xa000, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp73:
	.loc	1 326 32                        ; f16_gemm_gfx1250.py:326:32
	v_or3_b32 v250 /*v506*/, v251 /*v507*/, v250 /*v506*/, s1
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e32 vcc_lo, s9, v0
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp74:
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_dual_add_nc_u32 v253 /*v509*/, v252 /*v508*/, v170 /*v426*/ :: v_dual_lshrrev_b32 v178 /*v434*/, 4, v172 /*v428*/
.Ltmp75:
	.loc	1 328 25                        ; f16_gemm_gfx1250.py:328:25
	v_mul_lo_u32 v251 /*v507*/, v250 /*v506*/, s13
	.loc	1 329 33                        ; f16_gemm_gfx1250.py:329:33
	v_cmp_gt_i32_e64 s10, s8, v250 /*v506*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
.Ltmp76:
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_and_b32_e32 v254 /*v510*/, 0x9f0, v178 /*v434*/
	v_add_nc_u32_e32 v171 /*v427*/, v252 /*v508*/, v171 /*v427*/
.Ltmp77:
	.loc	1 329 39                        ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s10, vcc_lo
.Ltmp78:
	.loc	1 106 72                        ; f16_gemm_gfx1250.py:106:72 @[ f16_gemm_gfx1250.py:324:96 ]
	v_add_nc_u32_e32 v254 /*v510*/, v252 /*v508*/, v254 /*v510*/
	ds_load_b128 v[202:205] /*v[458:461]*/, v171 /*v427*/ offset:16384
	ds_load_b128 v[206:209] /*v[462:465]*/, v171 /*v427*/ offset:16416
	ds_load_b128 v[218:221] /*v[474:477]*/, v171 /*v427*/ offset:16448
	ds_load_b128 v[222:225] /*v[478:481]*/, v171 /*v427*/ offset:16480
	ds_load_b128 v[210:213] /*v[466:469]*/, v171 /*v427*/ offset:16512
	ds_load_b128 v[214:217] /*v[470:473]*/, v171 /*v427*/ offset:16544
	ds_load_b128 v[194:197] /*v[450:453]*/, v171 /*v427*/ offset:16576
	ds_load_b128 v[198:201] /*v[454:457]*/, v171 /*v427*/ offset:16608
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:24576
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:24608
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:512 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:528 ; 16-byte Folded Spill
	ds_load_b128 v[186:189] /*v[442:445]*/, v253 /*v509*/ offset:24640
	ds_load_b128 v[190:193] /*v[446:449]*/, v253 /*v509*/ offset:24672
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:24704
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:24736
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:544 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:560 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:24768
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:24800
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:480 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:496 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v254 /*v510*/ offset:32768
	ds_load_b128 v[174:177] /*v[430:433]*/, v254 /*v510*/ offset:32800
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:416 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:432 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v254 /*v510*/ offset:32832
	ds_load_b128 v[174:177] /*v[430:433]*/, v254 /*v510*/ offset:32864
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:448 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:464 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v254 /*v510*/ offset:32896
	ds_load_b128 v[174:177] /*v[430:433]*/, v254 /*v510*/ offset:32928
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:384 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:400 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v254 /*v510*/ offset:32960
	ds_load_b128 v[174:177] /*v[430:433]*/, v254 /*v510*/ offset:32992
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_or_b32_e32 v253 /*v509*/, 0xc000, v1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_lshrrev_b32_e32 v255 /*v511*/, 4, v255 /*v511*/
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:352 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:368 ; 16-byte Folded Spill
	v_lshrrev_b32_e32 v253 /*v509*/, 4, v253 /*v509*/
	v_and_b32_e32 v255 /*v511*/, 0xbf0, v255 /*v511*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v1, 0xe000, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_and_b32_e32 v253 /*v509*/, 0xdf0, v253 /*v509*/
	v_add_nc_u32_e32 v255 /*v511*/, v252 /*v508*/, v255 /*v511*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v1, 4, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u32_e32 v253 /*v509*/, v252 /*v508*/, v253 /*v509*/
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v255 /*v511*/ offset:40960
	ds_load_b128 v[174:177] /*v[430:433]*/, v255 /*v511*/ offset:40992
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:320 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:336 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v255 /*v511*/ offset:41024
	ds_load_b128 v[174:177] /*v[430:433]*/, v255 /*v511*/ offset:41056
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:288 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:304 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v255 /*v511*/ offset:41088
	ds_load_b128 v[174:177] /*v[430:433]*/, v255 /*v511*/ offset:41120
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:256 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:272 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v255 /*v511*/ offset:41152
	ds_load_b128 v[174:177] /*v[430:433]*/, v255 /*v511*/ offset:41184
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:224 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:240 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:49152
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:49184
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:160 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:176 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:49216
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:49248
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:128 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:144 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:49280
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:49312
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:96 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:112 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v253 /*v509*/ offset:49344
	ds_load_b128 v[174:177] /*v[430:433]*/, v253 /*v509*/ offset:49376
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v1, 0xff0, v1
	s_wait_dscnt 0x1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:64 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:80 ; 16-byte Folded Spill
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_nc_u32_e32 v1, v252 /*v508*/, v1
	s_wait_xcnt 0x0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[170:173] /*v[426:429]*/, v1 offset:57344
	ds_load_b128 v[174:177] /*v[430:433]*/, v1 offset:57376
	s_wait_dscnt 0x1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:32 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:48 ; 16-byte Folded Spill
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[178:181] /*v[434:437]*/, v1 offset:57408
	ds_load_b128 v[182:185] /*v[438:441]*/, v1 offset:57440
	s_wait_xcnt 0x0
	ds_load_b128 v[170:173] /*v[426:429]*/, v1 offset:57472
	ds_load_b128 v[174:177] /*v[430:433]*/, v1 offset:57504
	s_wait_dscnt 0x1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:16 ; 16-byte Folded Spill
	s_wait_xcnt 0x0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[170:173] /*v[426:429]*/, v1 offset:57536
	ds_load_b128 v[174:177] /*v[430:433]*/, v1 offset:57568
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp79:
	.loc	1 327 32                        ; f16_gemm_gfx1250.py:327:32
	v_or_b32_e32 v1, 4, v0
	s_wait_dscnt 0x1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:192 ; 16-byte Folded Spill
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:208 ; 16-byte Folded Spill
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, s1
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_9
; %bb.8:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[58:65] /*v[314:321]*/, v[162:169] /*v[418:425]*/, v[250:257]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_add_nc_u32 v252 /*v508*/, v251 /*v507*/, v0 :: v_dual_add_nc_u32 v254 /*v510*/, v251 /*v507*/, v1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v253 /*v509*/, 31, v252 /*v508*/ :: v_dual_ashrrev_i32 v255 /*v511*/, 31, v254 /*v510*/
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[66:73] /*v[322:329]*/, v[154:161] /*v[410:417]*/, v[250:257]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, 2, s[6:7]
	v_lshl_add_u64 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[250:257], v[50:57] /*v[306:313]*/, v[146:153] /*v[402:409]*/, v[250:257]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[250:257], v[42:49] /*v[298:305]*/, v[138:145] /*v[394:401]*/, v[250:257]
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[252:253] /*v[508:509]*/, v[250:253], off
	global_store_b128 v[254:255] /*v[510:511]*/, v[254:257], off
.LBB0_9:                                ; %.critedge
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v250, 32, v0
	v_or_b32_e32 v251, 36, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s0, s9, v250
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s2, s10, s0
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB0_11
; %bb.10:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[242:249], v[34:41] /*v[290:297]*/, v[162:169] /*v[418:425]*/, v[242:249]
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v252, v251 /*v507*/, v250 :: v_dual_add_nc_u32 v254, v251 /*v507*/, v251
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v253, 31, v252 :: v_dual_ashrrev_i32 v255, 31, v254
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[242:249], v[242:249] /*v[498:505]*/, v[154:161] /*v[410:417]*/, v[242:249]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[252:253], v[252:253], 2, s[6:7]
	v_lshl_add_u64 v[254:255], v[254:255], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[242:249], v[234:241] /*v[490:497]*/, v[146:153] /*v[402:409]*/, v[242:249]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[242:249], v[226:233] /*v[482:489]*/, v[138:145] /*v[394:401]*/, v[242:249]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[252:253], v[242:245], off
	global_store_b128 v[254:255], v[246:249], off
.LBB0_11:                               ; %.critedge2
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v242, 64, v0
	v_or_b32_e32 v243, 0x44, v0
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s1, s9, v242
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s3, s10, s1
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_mov_b32 s2, exec_lo
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[178:179] /*v[434:435]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[180:181] /*v[436:437]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[182:183] /*v[438:439]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[184:185] /*v[440:441]*/
	s_and_b32 s3, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 exec_lo, s3
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_13
; %bb.12:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[234:241], v[202:209] /*v[458:465]*/, v[162:169] /*v[418:425]*/, v[234:241]
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v244, v251 /*v507*/, v242 :: v_dual_add_nc_u32 v246, v251 /*v507*/, v243
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v245, 31, v244 :: v_dual_ashrrev_i32 v247, 31, v246
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[234:241], v[218:225] /*v[474:481]*/, v[154:161] /*v[410:417]*/, v[234:241]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[244:245], v[244:245], 2, s[6:7]
	v_lshl_add_u64 v[246:247], v[246:247], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[234:241], v[210:217] /*v[466:473]*/, v[146:153] /*v[402:409]*/, v[234:241]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[234:241], v[194:201] /*v[450:457]*/, v[138:145] /*v[394:401]*/, v[234:241]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[244:245], v[234:237], off
	global_store_b128 v[246:247], v[238:241], off
.LBB0_13:                               ; %.critedge4
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s2
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v234, 0x60, v0
	v_or_b32_e32 v235, 0x64, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s2, s9, v234
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s4, s10, s2
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB0_15
; %bb.14:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:512
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:528
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v236, v251 /*v507*/, v234 :: v_dual_add_nc_u32 v238, v251 /*v507*/, v235
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[226:233], v[178:185] /*v[434:441]*/, v[162:169] /*v[418:425]*/, v[226:233]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:544
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:560
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v237, 31, v236 :: v_dual_ashrrev_i32 v239, 31, v238
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[236:237], v[236:237], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[226:233], v[186:193] /*v[442:449]*/, v[154:161] /*v[410:417]*/, v[226:233]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u64 v[238:239], v[238:239], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[226:233], v[178:185] /*v[434:441]*/, v[146:153] /*v[402:409]*/, v[226:233]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:480
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:496
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[226:233], v[178:185] /*v[434:441]*/, v[138:145] /*v[394:401]*/, v[226:233]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[236:237], v[226:229], off
	global_store_b128 v[238:239], v[230:233], off
.LBB0_15:                               ; %.critedge6
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v226, 0x80, v0
	v_or_b32_e32 v227, 0x84, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s3, s9, v226
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s5, s10, s3
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB0_17
; %bb.16:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:416
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:432
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v228, v251 /*v507*/, v226 :: v_dual_add_nc_u32 v230, v251 /*v507*/, v227
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[218:225], v[178:185] /*v[434:441]*/, v[162:169] /*v[418:425]*/, v[218:225]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:448
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:464
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v229, 31, v228 :: v_dual_ashrrev_i32 v231, 31, v230
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u64 v[228:229], v[228:229], 2, s[6:7]
	v_lshl_add_u64 v[230:231], v[230:231], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[218:225], v[178:185] /*v[434:441]*/, v[154:161] /*v[410:417]*/, v[218:225]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:384
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:400
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[218:225], v[178:185] /*v[434:441]*/, v[146:153] /*v[402:409]*/, v[218:225]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:352
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:368
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[218:225], v[178:185] /*v[434:441]*/, v[138:145] /*v[394:401]*/, v[218:225]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[228:229], v[218:221], off
	global_store_b128 v[230:231], v[222:225], off
.LBB0_17:                               ; %.critedge8
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v218, 0xa0, v0
	v_or_b32_e32 v219, 0xa4, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s4, s9, v218
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s11, s10, s4
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s5, s11
	s_cbranch_execz .LBB0_19
; %bb.18:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:320
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:336
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v220, v251 /*v507*/, v218 :: v_dual_add_nc_u32 v222, v251 /*v507*/, v219
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[210:217], v[178:185] /*v[434:441]*/, v[162:169] /*v[418:425]*/, v[210:217]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:288
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:304
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v221, 31, v220 :: v_dual_ashrrev_i32 v223, 31, v222
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u64 v[220:221], v[220:221], 2, s[6:7]
	v_lshl_add_u64 v[222:223], v[222:223], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[210:217], v[178:185] /*v[434:441]*/, v[154:161] /*v[410:417]*/, v[210:217]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:256
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:272
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[210:217], v[178:185] /*v[434:441]*/, v[146:153] /*v[402:409]*/, v[210:217]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:224
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:240
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[210:217], v[178:185] /*v[434:441]*/, v[138:145] /*v[394:401]*/, v[210:217]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[220:221], v[210:213], off
	global_store_b128 v[222:223], v[214:217], off
.LBB0_19:                               ; %.critedge10
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s5
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v210, 0xc0, v0
	v_or_b32_e32 v211, 0xc4, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s5, s9, v210
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s12, s10, s5
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_21
; %bb.20:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:160
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:176
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v212, v251 /*v507*/, v210 :: v_dual_add_nc_u32 v214, v251 /*v507*/, v211
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[202:209], v[178:185] /*v[434:441]*/, v[162:169] /*v[418:425]*/, v[202:209]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:128
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:144
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v213, 31, v212 :: v_dual_ashrrev_i32 v215, 31, v214
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u64 v[212:213], v[212:213], 2, s[6:7]
	v_lshl_add_u64 v[214:215], v[214:215], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[202:209], v[178:185] /*v[434:441]*/, v[154:161] /*v[410:417]*/, v[202:209]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:96
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:112
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[202:209], v[178:185] /*v[434:441]*/, v[146:153] /*v[402:409]*/, v[202:209]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:64
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:80
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[202:209], v[178:185] /*v[434:441]*/, v[138:145] /*v[394:401]*/, v[202:209]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[212:213], v[202:205], off
	global_store_b128 v[214:215], v[206:209], off
.LBB0_21:                               ; %.critedge12
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s11
	.loc	1 327 32 is_stmt 1              ; f16_gemm_gfx1250.py:327:32
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v202, 0xe0, v0
	v_or_b32_e32 v203, 0xe4, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	.loc	1 329 58                        ; f16_gemm_gfx1250.py:329:58
	v_cmp_gt_i32_e64 s9, s9, v202
	.loc	1 329 39 is_stmt 0              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s11, s10, s9
	.loc	1 330 31 is_stmt 1              ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s10, s11
	s_cbranch_execz .LBB0_23
; %bb.22:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:32
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:48
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v204, v251 /*v507*/, v202 :: v_dual_add_nc_u32 v206, v251 /*v507*/, v203
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[194:201], v[178:185] /*v[434:441]*/, v[162:169] /*v[418:425]*/, v[194:201]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v205, 31, v204 :: v_dual_ashrrev_i32 v207, 31, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u64 v[204:205], v[204:205], 2, s[6:7]
	v_lshl_add_u64 v[206:207], v[206:207], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[194:201], v[170:177] /*v[426:433]*/, v[154:161] /*v[410:417]*/, v[194:201]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[154:157] /*v[410:413]*/, off, off
	scratch_load_b128 v[158:161] /*v[414:417]*/, off, off offset:16
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[194:201], v[154:161] /*v[410:417]*/, v[146:153] /*v[402:409]*/, v[194:201]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_clause 0x1
	scratch_load_b128 v[146:149] /*v[402:405]*/, off, off offset:192
	scratch_load_b128 v[150:153] /*v[406:409]*/, off, off offset:208
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[194:201], v[146:153] /*v[402:409]*/, v[138:145] /*v[394:401]*/, v[194:201]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[204:205], v[194:197], off
	global_store_b128 v[206:207], v[198:201], off
.LBB0_23:                               ; %.critedge14
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s10
	.loc	1 326 32 is_stmt 1              ; f16_gemm_gfx1250.py:326:32
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_or_b32_e32 v194, 64, v250 /*v506*/
	.loc	1 328 25                        ; f16_gemm_gfx1250.py:328:25
	s_lshl_b32 s11, s13, 6
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	.loc	1 329 33                        ; f16_gemm_gfx1250.py:329:33
	v_cmp_gt_i32_e64 s10, s8, v194
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 328 25                        ; f16_gemm_gfx1250.py:328:25
	v_add_nc_u32_e32 v194, s11, v251 /*v507*/
	.loc	1 329 39                        ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, vcc_lo
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_50
; %bb.24:                               ; %.critedge16
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_51
.LBB0_25:                               ; %.critedge18
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_52
.LBB0_26:                               ; %.critedge20
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_53
.LBB0_27:                               ; %.critedge22
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_54
.LBB0_28:                               ; %.critedge24
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_55
.LBB0_29:                               ; %.critedge26
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_56
.LBB0_30:                               ; %.critedge28
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s12, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s10, s12
	s_cbranch_execz .LBB0_32
.LBB0_31:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[138:141], off, off offset:32
	scratch_load_b128 v[142:145], off, off offset:48
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[130:137], v[138:145], v[130:137] /*v[386:393]*/, v[130:137]
	s_clause 0x1
	scratch_load_b128 v[142:145], off, off
	scratch_load_b128 v[146:149], off, off offset:16
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v138, v194, v202 :: v_dual_add_nc_u32 v140, v194, v203
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v139, 31, v138 :: v_dual_ashrrev_i32 v141, 31, v140
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[130:137], v[170:177] /*v[426:433]*/, v[122:129] /*v[378:385]*/, v[130:137]
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u64 v[138:139], v[138:139], 2, s[6:7]
	v_lshl_add_u64 v[140:141], v[140:141], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[130:137], v[142:149], v[114:121] /*v[370:377]*/, v[130:137]
	s_clause 0x1
	scratch_load_b128 v[142:145], off, off offset:192
	scratch_load_b128 v[146:149], off, off offset:208
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[130:137], v[142:149], v[106:113] /*v[362:369]*/, v[130:137]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[138:139], v[130:133], off
	global_store_b128 v[140:141], v[134:137], off
.LBB0_32:                               ; %.critedge30
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s10
	.loc	1 326 32 is_stmt 1              ; f16_gemm_gfx1250.py:326:32
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_or_b32_e32 v130, 0x80, v250 /*v506*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	1 329 33                        ; f16_gemm_gfx1250.py:329:33
	v_cmp_gt_i32_e64 s10, s8, v130
	.loc	1 328 25                        ; f16_gemm_gfx1250.py:328:25
	v_add_nc_u32_e32 v130, s11, v194
	.loc	1 329 39                        ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, vcc_lo
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_57
; %bb.33:                               ; %.critedge32
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_58
.LBB0_34:                               ; %.critedge34
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_59
.LBB0_35:                               ; %.critedge36
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_60
.LBB0_36:                               ; %.critedge38
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_61
.LBB0_37:                               ; %.critedge40
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_62
.LBB0_38:                               ; %.critedge42
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execnz .LBB0_63
.LBB0_39:                               ; %.critedge44
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s12, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s10, s12
	s_cbranch_execz .LBB0_41
.LBB0_40:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[74:77], off, off offset:32
	scratch_load_b128 v[78:81], off, off offset:48
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[66:73], v[74:81], v[82:89] /*v[338:345]*/, v[66:73]
	s_clause 0x1
	scratch_load_b128 v[78:81], off, off
	scratch_load_b128 v[82:85], off, off offset:16
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v74, v130, v202 :: v_dual_add_nc_u32 v76, v130, v203
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v75, 31, v74 :: v_dual_ashrrev_i32 v77, 31, v76
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[66:73], v[170:177] /*v[426:433]*/, v[98:105] /*v[354:361]*/, v[66:73]
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u64 v[74:75], v[74:75], 2, s[6:7]
	v_lshl_add_u64 v[76:77], v[76:77], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[66:73], v[78:85], v[90:97] /*v[346:353]*/, v[66:73]
	s_clause 0x1
	scratch_load_b128 v[78:81], off, off offset:192
	scratch_load_b128 v[82:85], off, off offset:208
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[66:73], v[78:85], v[74:81] /*v[330:337]*/, v[66:73]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[74:75], v[66:69], off
	global_store_b128 v[76:77], v[70:73], off
.LBB0_41:                               ; %.critedge46
	.loc	1 0 31                          ; f16_gemm_gfx1250.py:0:31
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s10
	.loc	1 326 32 is_stmt 1              ; f16_gemm_gfx1250.py:326:32
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_or_b32_e32 v66, 0xc0, v250 /*v506*/
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	1 329 33                        ; f16_gemm_gfx1250.py:329:33
	v_cmp_gt_i32_e64 s8, s8, v66
	.loc	1 328 25                        ; f16_gemm_gfx1250.py:328:25
	v_add_nc_u32_e32 v66, s11, v130
	.loc	1 329 39                        ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s11, s8, vcc_lo
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s10, s11
	s_cbranch_execnz .LBB0_64
; %bb.42:                               ; %.critedge48
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s10
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s10, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s10
	s_cbranch_execnz .LBB0_65
.LBB0_43:                               ; %.critedge50
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_66
.LBB0_44:                               ; %.critedge52
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_67
.LBB0_45:                               ; %.critedge54
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_68
.LBB0_46:                               ; %.critedge56
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_69
.LBB0_47:                               ; %.critedge58
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_70
.LBB0_48:                               ; %.critedge60
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s0, s8, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s1, s0
	s_cbranch_execnz .LBB0_71
.LBB0_49:                               ; %.critedge62
	.loc	1 330 4 is_stmt 0               ; f16_gemm_gfx1250.py:330:4
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB0_50:
	.loc	1 0 4                           ; f16_gemm_gfx1250.py:0:4
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[58:65] /*v[314:321]*/, v[130:137] /*v[386:393]*/, v[186:193]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v196, v194, v0 :: v_dual_add_nc_u32 v198, v194, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v197, 31, v196 :: v_dual_ashrrev_i32 v199, 31, v198
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[66:73] /*v[322:329]*/, v[122:129] /*v[378:385]*/, v[186:193]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[196:197], v[196:197], 2, s[6:7]
	v_lshl_add_u64 v[198:199], v[198:199], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[186:193], v[50:57] /*v[306:313]*/, v[114:121] /*v[370:377]*/, v[186:193]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[186:193], v[42:49] /*v[298:305]*/, v[106:113] /*v[362:369]*/, v[186:193]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[196:197], v[186:189], off
	global_store_b128 v[198:199], v[190:193], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_25
.LBB0_51:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[178:185], v[34:41] /*v[290:297]*/, v[130:137] /*v[386:393]*/, v[178:185]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v186, v194, v250 :: v_dual_add_nc_u32 v188, v194, v251
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v187, 31, v186 :: v_dual_ashrrev_i32 v189, 31, v188
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[178:185], v[242:249] /*v[498:505]*/, v[122:129] /*v[378:385]*/, v[178:185]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[186:187], v[186:187], 2, s[6:7]
	v_lshl_add_u64 v[188:189], v[188:189], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[178:185], v[234:241] /*v[490:497]*/, v[114:121] /*v[370:377]*/, v[178:185]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[178:185], v[226:233] /*v[482:489]*/, v[106:113] /*v[362:369]*/, v[178:185]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[186:187], v[178:181], off
	global_store_b128 v[188:189], v[182:185], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_26
.LBB0_52:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[170:177], v[202:209] /*v[458:465]*/, v[130:137] /*v[386:393]*/, v[170:177]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v178, v194, v242 :: v_dual_add_nc_u32 v180, v194, v243
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v179, 31, v178 :: v_dual_ashrrev_i32 v181, 31, v180
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[170:177], v[218:225] /*v[474:481]*/, v[122:129] /*v[378:385]*/, v[170:177]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[178:179], v[178:179], 2, s[6:7]
	v_lshl_add_u64 v[180:181], v[180:181], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[170:177], v[210:217] /*v[466:473]*/, v[114:121] /*v[370:377]*/, v[170:177]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[170:177], v[194:201] /*v[450:457]*/, v[106:113] /*v[362:369]*/, v[170:177]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[178:179], v[170:173], off
	global_store_b128 v[180:181], v[174:177], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_27
.LBB0_53:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[170:173], off, off offset:512
	scratch_load_b128 v[174:177], off, off offset:528
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[162:169], v[170:177], v[130:137] /*v[386:393]*/, v[162:169]
	s_clause 0x1
	scratch_load_b128 v[174:177], off, off offset:544
	scratch_load_b128 v[178:181], off, off offset:560
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v170, v194, v234 :: v_dual_add_nc_u32 v172, v194, v235
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v171, 31, v170 :: v_dual_ashrrev_i32 v173, 31, v172
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[162:169], v[186:193] /*v[442:449]*/, v[122:129] /*v[378:385]*/, v[162:169]
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u64 v[170:171], v[170:171], 2, s[6:7]
	v_lshl_add_u64 v[172:173], v[172:173], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[162:169], v[174:181], v[114:121] /*v[370:377]*/, v[162:169]
	s_clause 0x1
	scratch_load_b128 v[174:177], off, off offset:480
	scratch_load_b128 v[178:181], off, off offset:496
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[162:169], v[174:181], v[106:113] /*v[362:369]*/, v[162:169]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[170:171], v[162:165], off
	global_store_b128 v[172:173], v[166:169], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_28
.LBB0_54:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[162:165], off, off offset:416
	scratch_load_b128 v[166:169], off, off offset:432
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[154:161], v[162:169], v[130:137] /*v[386:393]*/, v[154:161]
	s_clause 0x1
	scratch_load_b128 v[162:165], off, off offset:448
	scratch_load_b128 v[166:169], off, off offset:464
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[154:161], v[162:169], v[122:129] /*v[378:385]*/, v[154:161]
	s_clause 0x1
	scratch_load_b128 v[166:169], off, off offset:384
	scratch_load_b128 v[170:173], off, off offset:400
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v162, v194, v226 :: v_dual_add_nc_u32 v164, v194, v227
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v163, 31, v162 :: v_dual_ashrrev_i32 v165, 31, v164
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[154:161], v[166:173], v[114:121] /*v[370:377]*/, v[154:161]
	s_clause 0x1
	scratch_load_b128 v[166:169], off, off offset:352
	scratch_load_b128 v[170:173], off, off offset:368
	v_nop
	v_lshl_add_u64 v[162:163], v[162:163], 2, s[6:7]
	v_lshl_add_u64 v[164:165], v[164:165], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[154:161], v[166:173], v[106:113] /*v[362:369]*/, v[154:161]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[162:163], v[154:157], off
	global_store_b128 v[164:165], v[158:161], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_29
.LBB0_55:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[154:157], off, off offset:320
	scratch_load_b128 v[158:161], off, off offset:336
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[146:153], v[154:161], v[130:137] /*v[386:393]*/, v[146:153]
	s_clause 0x1
	scratch_load_b128 v[154:157], off, off offset:288
	scratch_load_b128 v[158:161], off, off offset:304
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[146:153], v[154:161], v[122:129] /*v[378:385]*/, v[146:153]
	s_clause 0x1
	scratch_load_b128 v[158:161], off, off offset:256
	scratch_load_b128 v[162:165], off, off offset:272
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v154, v194, v218 :: v_dual_add_nc_u32 v156, v194, v219
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v155, 31, v154 :: v_dual_ashrrev_i32 v157, 31, v156
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[146:153], v[158:165], v[114:121] /*v[370:377]*/, v[146:153]
	s_clause 0x1
	scratch_load_b128 v[158:161], off, off offset:224
	scratch_load_b128 v[162:165], off, off offset:240
	v_nop
	v_lshl_add_u64 v[154:155], v[154:155], 2, s[6:7]
	v_lshl_add_u64 v[156:157], v[156:157], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[146:153], v[158:165], v[106:113] /*v[362:369]*/, v[146:153]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[154:155], v[146:149], off
	global_store_b128 v[156:157], v[150:153], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_30
.LBB0_56:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[146:149], off, off offset:160
	scratch_load_b128 v[150:153], off, off offset:176
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[138:145], v[146:153], v[130:137] /*v[386:393]*/, v[138:145]
	s_clause 0x1
	scratch_load_b128 v[146:149], off, off offset:128
	scratch_load_b128 v[150:153], off, off offset:144
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[138:145], v[146:153], v[122:129] /*v[378:385]*/, v[138:145]
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:96
	scratch_load_b128 v[154:157], off, off offset:112
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v146, v194, v210 :: v_dual_add_nc_u32 v148, v194, v211
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v147, 31, v146 :: v_dual_ashrrev_i32 v149, 31, v148
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[138:145], v[150:157], v[114:121] /*v[370:377]*/, v[138:145]
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:64
	scratch_load_b128 v[154:157], off, off offset:80
	v_nop
	v_lshl_add_u64 v[146:147], v[146:147], 2, s[6:7]
	v_lshl_add_u64 v[148:149], v[148:149], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[138:145], v[150:157], v[106:113] /*v[362:369]*/, v[138:145]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[146:147], v[138:141], off
	global_store_b128 v[148:149], v[142:145], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s12, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s10, s12
	s_cbranch_execnz .LBB0_31
	s_branch .LBB0_32
.LBB0_57:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[58:65] /*v[314:321]*/, v[82:89] /*v[338:345]*/, v[122:129]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v132, v130, v0 :: v_dual_add_nc_u32 v134, v130, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v133, 31, v132 :: v_dual_ashrrev_i32 v135, 31, v134
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[66:73] /*v[322:329]*/, v[98:105] /*v[354:361]*/, v[122:129]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[132:133], v[132:133], 2, s[6:7]
	v_lshl_add_u64 v[134:135], v[134:135], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[122:129], v[50:57] /*v[306:313]*/, v[90:97] /*v[346:353]*/, v[122:129]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[122:129], v[42:49] /*v[298:305]*/, v[74:81] /*v[330:337]*/, v[122:129]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[132:133], v[122:125], off
	global_store_b128 v[134:135], v[126:129], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_34
.LBB0_58:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[114:121], v[34:41] /*v[290:297]*/, v[82:89] /*v[338:345]*/, v[114:121]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v122, v130, v250 :: v_dual_add_nc_u32 v124, v130, v251
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v123, 31, v122 :: v_dual_ashrrev_i32 v125, 31, v124
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[114:121], v[242:249] /*v[498:505]*/, v[98:105] /*v[354:361]*/, v[114:121]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[122:123], v[122:123], 2, s[6:7]
	v_lshl_add_u64 v[124:125], v[124:125], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[114:121], v[234:241] /*v[490:497]*/, v[90:97] /*v[346:353]*/, v[114:121]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[114:121], v[226:233] /*v[482:489]*/, v[74:81] /*v[330:337]*/, v[114:121]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[122:123], v[114:117], off
	global_store_b128 v[124:125], v[118:121], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_35
.LBB0_59:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[106:113], v[202:209] /*v[458:465]*/, v[82:89] /*v[338:345]*/, v[106:113]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v114, v130, v242 :: v_dual_add_nc_u32 v116, v130, v243
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v115, 31, v114 :: v_dual_ashrrev_i32 v117, 31, v116
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[106:113], v[218:225] /*v[474:481]*/, v[98:105] /*v[354:361]*/, v[106:113]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[114:115], v[114:115], 2, s[6:7]
	v_lshl_add_u64 v[116:117], v[116:117], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[106:113], v[210:217] /*v[466:473]*/, v[90:97] /*v[346:353]*/, v[106:113]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[106:113], v[194:201] /*v[450:457]*/, v[74:81] /*v[330:337]*/, v[106:113]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[114:115], v[106:109], off
	global_store_b128 v[116:117], v[110:113], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_36
.LBB0_60:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[106:109], off, off offset:512
	scratch_load_b128 v[110:113], off, off offset:528
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[98:105], v[106:113], v[82:89] /*v[338:345]*/, v[98:105]
	s_clause 0x1
	scratch_load_b128 v[110:113], off, off offset:544
	scratch_load_b128 v[114:117], off, off offset:560
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v106, v130, v234 :: v_dual_add_nc_u32 v108, v130, v235
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v107, 31, v106 :: v_dual_ashrrev_i32 v109, 31, v108
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[98:105], v[186:193] /*v[442:449]*/, v[98:105] /*v[354:361]*/, v[98:105]
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u64 v[106:107], v[106:107], 2, s[6:7]
	v_lshl_add_u64 v[108:109], v[108:109], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[98:105], v[110:117], v[90:97] /*v[346:353]*/, v[98:105]
	s_clause 0x1
	scratch_load_b128 v[110:113], off, off offset:480
	scratch_load_b128 v[114:117], off, off offset:496
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[98:105], v[110:117], v[74:81] /*v[330:337]*/, v[98:105]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[106:107], v[98:101], off
	global_store_b128 v[108:109], v[102:105], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_37
.LBB0_61:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[98:101], off, off offset:416
	scratch_load_b128 v[102:105], off, off offset:432
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[90:97], v[98:105], v[82:89] /*v[338:345]*/, v[90:97]
	s_clause 0x1
	scratch_load_b128 v[98:101], off, off offset:448
	scratch_load_b128 v[102:105], off, off offset:464
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[90:97], v[98:105], v[98:105] /*v[354:361]*/, v[90:97]
	s_clause 0x1
	scratch_load_b128 v[102:105], off, off offset:384
	scratch_load_b128 v[106:109], off, off offset:400
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v98, v130, v226 :: v_dual_add_nc_u32 v100, v130, v227
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v99, 31, v98 :: v_dual_ashrrev_i32 v101, 31, v100
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[90:97], v[102:109], v[90:97] /*v[346:353]*/, v[90:97]
	s_clause 0x1
	scratch_load_b128 v[102:105], off, off offset:352
	scratch_load_b128 v[106:109], off, off offset:368
	v_nop
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[6:7]
	v_lshl_add_u64 v[100:101], v[100:101], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[90:97], v[102:109], v[74:81] /*v[330:337]*/, v[90:97]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[98:99], v[90:93], off
	global_store_b128 v[100:101], v[94:97], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_38
.LBB0_62:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[90:93], off, off offset:320
	scratch_load_b128 v[94:97], off, off offset:336
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[82:89], v[90:97], v[82:89] /*v[338:345]*/, v[82:89]
	s_clause 0x1
	scratch_load_b128 v[90:93], off, off offset:288
	scratch_load_b128 v[94:97], off, off offset:304
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[82:89], v[90:97], v[98:105] /*v[354:361]*/, v[82:89]
	s_clause 0x1
	scratch_load_b128 v[94:97], off, off offset:256
	scratch_load_b128 v[98:101], off, off offset:272
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v90, v130, v218 :: v_dual_add_nc_u32 v92, v130, v219
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v91, 31, v90 :: v_dual_ashrrev_i32 v93, 31, v92
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[82:89], v[94:101], v[90:97] /*v[346:353]*/, v[82:89]
	s_clause 0x1
	scratch_load_b128 v[94:97], off, off offset:224
	scratch_load_b128 v[98:101], off, off offset:240
	v_nop
	v_lshl_add_u64 v[90:91], v[90:91], 2, s[6:7]
	v_lshl_add_u64 v[92:93], v[92:93], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[82:89], v[94:101], v[74:81] /*v[330:337]*/, v[82:89]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[90:91], v[82:85], off
	global_store_b128 v[92:93], v[86:89], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s13, s10, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s12, s13
	s_cbranch_execz .LBB0_39
.LBB0_63:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[82:85], off, off offset:160
	scratch_load_b128 v[86:89], off, off offset:176
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[74:81], v[82:89], v[82:89] /*v[338:345]*/, v[74:81]
	s_clause 0x1
	scratch_load_b128 v[82:85], off, off offset:128
	scratch_load_b128 v[86:89], off, off offset:144
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[74:81], v[82:89], v[98:105] /*v[354:361]*/, v[74:81]
	s_clause 0x1
	scratch_load_b128 v[86:89], off, off offset:96
	scratch_load_b128 v[90:93], off, off offset:112
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v82, v130, v210 :: v_dual_add_nc_u32 v84, v130, v211
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_dual_ashrrev_i32 v83, 31, v82 :: v_dual_ashrrev_i32 v85, 31, v84
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[74:81], v[86:93], v[90:97] /*v[346:353]*/, v[74:81]
	s_clause 0x1
	scratch_load_b128 v[86:89], off, off offset:64
	scratch_load_b128 v[90:93], off, off offset:80
	v_nop
	v_lshl_add_u64 v[82:83], v[82:83], 2, s[6:7]
	v_lshl_add_u64 v[84:85], v[84:85], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[74:81], v[86:93], v[74:81] /*v[330:337]*/, v[74:81]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[82:83], v[74:77], off
	global_store_b128 v[84:85], v[78:81], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s12
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s12, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s10, s12
	s_cbranch_execnz .LBB0_40
	s_branch .LBB0_41
.LBB0_64:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[58:65], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[58:65]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v0, v66, v0 :: v_dual_add_nc_u32 v68, v66, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_ashrrev_i32_e32 v1, 31, v0
	v_ashrrev_i32_e32 v69, 31, v68
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[58:65], v[66:73] /*v[322:329]*/, v[26:33] /*v[282:289]*/, v[58:65]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[68:69], v[68:69], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[58:65], v[50:57] /*v[306:313]*/, v[18:25] /*v[274:281]*/, v[58:65]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[58:65], v[42:49] /*v[298:305]*/, v[10:17] /*v[266:273]*/, v[58:65]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[58:61], off
	global_store_b128 v[68:69], v[62:65], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s10
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s10, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s10
	s_cbranch_execz .LBB0_43
.LBB0_65:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[50:57], v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[50:57]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v0, v66, v250 :: v_dual_add_nc_u32 v58, v66, v251
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v1, 31, v0 :: v_dual_ashrrev_i32 v59, 31, v58
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[50:57], v[242:249] /*v[498:505]*/, v[26:33] /*v[282:289]*/, v[50:57]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[50:57], v[234:241] /*v[490:497]*/, v[18:25] /*v[274:281]*/, v[50:57]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[50:57], v[226:233] /*v[482:489]*/, v[10:17] /*v[266:273]*/, v[50:57]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[50:53], off
	global_store_b128 v[58:59], v[54:57], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_44
.LBB0_66:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[42:49], v[202:209] /*v[458:465]*/, v[2:9] /*v[258:265]*/, v[42:49]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_dual_add_nc_u32 v0, v66, v242 :: v_dual_add_nc_u32 v50, v66, v243
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_ashrrev_i32 v1, 31, v0 :: v_dual_ashrrev_i32 v51, 31, v50
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[42:49], v[218:225] /*v[474:481]*/, v[26:33] /*v[282:289]*/, v[42:49]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[50:51], v[50:51], 2, s[6:7]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[42:49], v[210:217] /*v[466:473]*/, v[18:25] /*v[274:281]*/, v[42:49]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[42:49], v[194:201] /*v[450:457]*/, v[10:17] /*v[266:273]*/, v[42:49]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[42:45], off
	global_store_b128 v[50:51], v[46:49], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_45
.LBB0_67:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[42:45], off, off offset:512 th:TH_LOAD_LU
	scratch_load_b128 v[46:49], off, off offset:528 th:TH_LOAD_LU
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_add_nc_u32_e32 v0, v66, v234
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[34:41], v[42:49], v[2:9] /*v[258:265]*/, v[34:41]
	s_clause 0x1
	scratch_load_b128 v[44:47], off, off offset:544 th:TH_LOAD_LU
	scratch_load_b128 v[48:51], off, off offset:560 th:TH_LOAD_LU
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v42, v66, v235 :: v_dual_ashrrev_i32 v1, 31, v0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v43, 31, v42
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[34:41], v[186:193] /*v[442:449]*/, v[26:33] /*v[282:289]*/, v[34:41]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[34:41], v[44:51], v[18:25] /*v[274:281]*/, v[34:41]
	s_clause 0x1
	scratch_load_b128 v[44:47], off, off offset:480 th:TH_LOAD_LU
	scratch_load_b128 v[48:51], off, off offset:496 th:TH_LOAD_LU
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[34:41], v[44:51], v[10:17] /*v[266:273]*/, v[34:41]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[34:37], off
	global_store_b128 v[42:43], v[38:41], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_46
.LBB0_68:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:416 th:TH_LOAD_LU
	scratch_load_b128 v[38:41], off, off offset:432 th:TH_LOAD_LU
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_add_nc_u32_e32 v0, v66, v226
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[26:33], v[34:41], v[2:9] /*v[258:265]*/, v[26:33]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:448 th:TH_LOAD_LU
	scratch_load_b128 v[38:41], off, off offset:464 th:TH_LOAD_LU
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[26:33], v[34:41], v[26:33] /*v[282:289]*/, v[26:33]
	s_clause 0x1
	scratch_load_b128 v[36:39], off, off offset:384 th:TH_LOAD_LU
	scratch_load_b128 v[40:43], off, off offset:400 th:TH_LOAD_LU
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v34, v66, v227 :: v_dual_ashrrev_i32 v1, 31, v0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[26:33], v[36:43], v[18:25] /*v[274:281]*/, v[26:33]
	s_clause 0x1
	scratch_load_b128 v[36:39], off, off offset:352 th:TH_LOAD_LU
	scratch_load_b128 v[40:43], off, off offset:368 th:TH_LOAD_LU
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[26:33], v[36:43], v[10:17] /*v[266:273]*/, v[26:33]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[26:29], off
	global_store_b128 v[34:35], v[30:33], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_47
.LBB0_69:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[26:29], off, off offset:320 th:TH_LOAD_LU
	scratch_load_b128 v[30:33], off, off offset:336 th:TH_LOAD_LU
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_add_nc_u32_e32 v0, v66, v218
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[18:25], v[26:33], v[2:9] /*v[258:265]*/, v[18:25]
	s_clause 0x1
	scratch_load_b128 v[26:29], off, off offset:288 th:TH_LOAD_LU
	scratch_load_b128 v[30:33], off, off offset:304 th:TH_LOAD_LU
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[18:25], v[26:33], v[26:33] /*v[282:289]*/, v[18:25]
	s_clause 0x1
	scratch_load_b128 v[28:31], off, off offset:256 th:TH_LOAD_LU
	scratch_load_b128 v[32:35], off, off offset:272 th:TH_LOAD_LU
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v26, v66, v219 :: v_dual_ashrrev_i32 v1, 31, v0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[18:25], v[28:35], v[18:25] /*v[274:281]*/, v[18:25]
	s_clause 0x1
	scratch_load_b128 v[28:31], off, off offset:224 th:TH_LOAD_LU
	scratch_load_b128 v[32:35], off, off offset:240 th:TH_LOAD_LU
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[18:25], v[28:35], v[10:17] /*v[266:273]*/, v[18:25]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[18:21], off
	global_store_b128 v[26:27], v[22:25], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s1, s8, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_48
.LBB0_70:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:160 th:TH_LOAD_LU
	scratch_load_b128 v[22:25], off, off offset:176 th:TH_LOAD_LU
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_add_nc_u32_e32 v0, v66, v210
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[10:17], v[18:25], v[2:9] /*v[258:265]*/, v[10:17]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:128 th:TH_LOAD_LU
	scratch_load_b128 v[22:25], off, off offset:144 th:TH_LOAD_LU
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[10:17], v[18:25], v[26:33] /*v[282:289]*/, v[10:17]
	s_clause 0x1
	scratch_load_b128 v[20:23], off, off offset:96 th:TH_LOAD_LU
	scratch_load_b128 v[24:27], off, off offset:112 th:TH_LOAD_LU
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v18, v66, v211 :: v_dual_ashrrev_i32 v1, 31, v0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[10:17], v[20:27], v[18:25] /*v[274:281]*/, v[10:17]
	s_clause 0x1
	scratch_load_b128 v[20:23], off, off offset:64 th:TH_LOAD_LU
	scratch_load_b128 v[24:27], off, off offset:80 th:TH_LOAD_LU
	v_lshl_add_u64 v[18:19], v[18:19], 2, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[10:17], v[20:27], v[10:17] /*v[266:273]*/, v[10:17]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[10:13], off
	global_store_b128 v[18:19], v[14:17], off
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 329 39 is_stmt 1              ; f16_gemm_gfx1250.py:329:39
	s_and_b32 s0, s8, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 330 31                        ; f16_gemm_gfx1250.py:330:31
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_49
.LBB0_71:
	.loc	1 0 31 is_stmt 0                ; f16_gemm_gfx1250.py:0:31
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:32 th:TH_LOAD_LU
	scratch_load_b128 v[14:17], off, off offset:48 th:TH_LOAD_LU
	.loc	1 328 44 is_stmt 1              ; f16_gemm_gfx1250.py:328:44
	v_add_nc_u32_e32 v0, v66, v202
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[2:9], v[10:17], v[2:9] /*v[258:265]*/, v[2:9]
	s_clause 0x1
	scratch_load_b128 v[12:15], off, off th:TH_LOAD_LU
	scratch_load_b128 v[16:19], off, off offset:16 th:TH_LOAD_LU
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v10, v66, v203 :: v_dual_ashrrev_i32 v1, 31, v0
	.loc	1 330 23                        ; f16_gemm_gfx1250.py:330:23
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v11, 31, v10
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_f16 v[2:9], v[170:177] /*v[426:433]*/, v[26:33] /*v[282:289]*/, v[2:9]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_f16 v[2:9], v[12:19], v[18:25] /*v[274:281]*/, v[2:9]
	s_clause 0x1
	scratch_load_b128 v[12:15], off, off offset:192 th:TH_LOAD_LU
	scratch_load_b128 v[16:19], off, off offset:208 th:TH_LOAD_LU
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_f16 v[2:9], v[12:19], v[10:17] /*v[266:273]*/, v[2:9]
	s_set_vgpr_msb 0                        ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 330 31 is_stmt 0              ; f16_gemm_gfx1250.py:330:31
	s_clause 0x1
	global_store_b128 v[0:1], v[2:5], off
	global_store_b128 v[10:11], v[6:9], off
	.loc	1 330 4                         ; f16_gemm_gfx1250.py:330:4
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp80:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_tdm_pipelined_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 580
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 18
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 16
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 52
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 131
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	gemm_tdm_pipelined_kernel, .Lfunc_end0-gemm_tdm_pipelined_kernel
	.cfi_endproc
                                        ; -- End function
	.set gemm_tdm_pipelined_kernel.num_vgpr, 512
	.set gemm_tdm_pipelined_kernel.num_agpr, 0
	.set gemm_tdm_pipelined_kernel.numbered_sgpr, 52
	.set gemm_tdm_pipelined_kernel.num_named_barrier, 0
	.set gemm_tdm_pipelined_kernel.private_seg_size, 580
	.set gemm_tdm_pipelined_kernel.uses_vcc, 1
	.set gemm_tdm_pipelined_kernel.uses_flat_scratch, 1
	.set gemm_tdm_pipelined_kernel.has_dyn_sized_stack, 0
	.set gemm_tdm_pipelined_kernel.has_recursion, 0
	.set gemm_tdm_pipelined_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16664
; TotalNumSgprs: 54
; NumVgprs: 512
; ScratchSize: 580
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 512
; NamedBarCnt: 0
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 18
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x93 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x6d DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	315                             ; DW_AT_call_line
	.byte	105                             ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	300                             ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	306                             ; DW_AT_call_line
	.byte	100                             ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x68:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	317                             ; DW_AT_call_line
	.byte	35                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x75:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	324                             ; DW_AT_call_line
	.byte	96                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x82:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	318                             ; DW_AT_call_line
	.byte	105                             ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x8f:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	320                             ; DW_AT_call_line
	.byte	92                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"f16_gemm_gfx1250.py"           ; string offset=7
.Linfo_string2:
	.asciz	"/home/jung/jp/triton/third_party/amd/python/examples/gluon" ; string offset=27
.Linfo_string3:
	.asciz	"gemm_tdm_pipelined_kernel"     ; string offset=86
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 256
    .name:           gemm_tdm_pipelined_kernel
    .private_segment_fixed_size: 580
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         gemm_tdm_pipelined_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 144
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:

