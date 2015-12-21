/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE

 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef __NVRTC_H__
#define __NVRTC_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>


/*****************************//**
 *
 * \defgroup error Error Handling
 *
 ********************************/


/**
 * \ingroup error
 * \brief   CUDA Online Compiler API call result code.
 */
typedef enum {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
} nvrtcResult;


/**
 * \ingroup error
 * \brief   ::nvrtcGetErrorString is a helper function that stringifies the
 *          given #nvrtcResult code, e.g., \link #nvrtcResult NVRTC_SUCCESS
 *          \endlink to \c "NVRTC_SUCCESS".  For unrecognized enumeration
 *          values, it returns \c "NVRTC_ERROR unknown".
 *
 * \param   [in] result CUDA Online Compiler API result code.
 * \return  Message string for the given #nvrtcResult code.
 */
const char *nvrtcGetErrorString(nvrtcResult result);


/****************************************//**
 *
 * \defgroup query General Information Query
 *
 *******************************************/


/**
 * \ingroup query
 * \brief   ::nvrtcVersion sets the output parameters \p major and \p minor
 *          with the CUDA Online Compiler version number.
 *
 * \param   [out] major CUDA Online Compiler major version number.
 * \param   [out] minor CUDA Online Compiler minor version number.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 */
nvrtcResult nvrtcVersion(int *major, int *minor);


/********************************//**
 *
 * \defgroup compilation Compilation
 *
 ***********************************/


/**
 * \ingroup compilation
 * \brief   ::nvrtcProgram is the unit of compilation, and an opaque handle for
 *          a program.
 *
 * To compile a CUDA program string, an instance of nvrtcProgram must be
 * created first with ::nvrtcCreateProgram, then compiled with
 * ::nvrtcCompileProgram.
 */
typedef struct _nvrtcProgram *nvrtcProgram;


/**
 * \ingroup compilation
 * \brief   ::nvrtcCreateProgram creates an instance of ::nvrtcProgram with the
 *          given input parameters, and sets the output parameter \p prog with
 *          it.
 *
 * \param   [out] prog         CUDA Online Compiler program.
 * \param   [in]  src          CUDA program source.
 * \param   [in]  name         CUDA program name.\n
 *                             \p name can be \c NULL; \c "default_program" is
 *                             used when \p name is \c NULL.
 * \param   [in]  numHeaders   Number of headers used.\n
 *                             \p numHeaders must be greater than or equal to 0.
 * \param   [in]  headers      Sources of the headers.\n
 *                             \p headers can be \c NULL when \p numHeaders is
 *                             0.
 * \param   [in]  includeNames Name of each header by which they can be
 *                             included in the CUDA program source.\n
 *                             \p includeNames can be \c NULL when \p numHeaders
 *                             is 0.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_PROGRAM_CREATION_FAILURE \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcDestroyProgram
 */
nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char **headers,
                               const char **includeNames);


/**
 * \ingroup compilation
 * \brief   ::nvrtcDestroyProgram destroys the given program.
 *
 * \param    [in] prog CUDA Online Compiler program.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcCreateProgram
 */
nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);


/**
 * \ingroup compilation
 * \brief   ::nvrtcCompileProgram compiles the given program.
 *
 * The valid compiler options are:
 *
 *   - Compilation targets
 *     - --gpu-architecture=<em>\<GPU architecture name\></em> (-arch)\n
 *       Specify the name of the class of GPU architectures for which the
 *       input must be compiled.\n
 *       - Valid <em>GPU architecture name</em>s:
 *         - compute_20
 *         - compute_30
 *         - compute_35
 *         - compute_50
 *       - Default: compute_20
 *   - Separate compilation / whole-program compilation
 *     - --device-c (-dc)\n
 *       Generate relocatable code that can be linked with other relocatable
 *       device code.  It is equivalent to --relocatable-device-code=true.
 *     - --device-w (-dw)\n
 *       Generate non-relocatable code.  It is equivalent to
 *       --relocatable-device-code=false.
 *     - --relocatable-device-code=<em>[true, false]</em> (-rdc)\n
 *       Enable (disable) the generation of relocatable device code.
 *       - Default: false
 *   - Debugging support
 *     - --device-debug (-G)\n
 *       Generate debug information.
 *     - --generate-line-info (-lineinfo)\n
 *       Generate line-number information.
 *   - Code generation
 *     - --maxrregcount=<em>\<N\></em> (-maxrregcount)\n
 *       Specify the maximum amount of registers that GPU functions can use.
 *       Until a function-specific limit, a higher value will generally
 *       increase the performance of individual GPU threads that execute this
 *       function.  However, because thread registers are allocated from a
 *       global register pool on each GPU, a higher value of this option will
 *       also reduce the maximum thread block size, thereby reducing the amount
 *       of thread parallelism.  Hence, a good maxrregcount value is the result
 *       of a trade-off.  If this option is not specified, then no maximum is
 *       assumed.  Value less than the minimum registers required by ABI will
 *       be bumped up by the compiler to ABI minimum limit.
 *     - --ftz=<em>[true, false]</em> (-ftz)\n
 *       When performing single-precision floating-point operations, flush
 *       denormal values to zero or preserve denormal values.  --use_fast_math
 *       implies --ftz=true.
 *       - Default: false
 *     - --prec-sqrt=<em>[true, false]</em> (-prec-sqrt)\n
 *       For single-precision floating-point square root, use IEEE
 *       round-to-nearest mode or use a faster approximation.  --use_fast_math
 *       implies --prec-sqrt=false.
 *       - Default: true
 *     - --prec-div=<em>[true, false]</em> (-prec-div)\n
 *       For single-precision floating-point division and reciprocals, use IEEE
 *       round-to-nearest mode or use a faster approximation.  --use_fast_math
 *       implies --prec-div=false.
 *       - Default: true
 *     - --fmad=<em>[true, false]</em> (-fmad)\n
 *       Enables (disables) the contraction of floating-point multiplies and
 *       adds/subtracts into floating-point multiply-add operations (FMAD,
 *       FFMA, or DFMA).  --use_fast_math implies --fmad=true.
 *       - Default: true
 *     - --use_fast_math (-use_fast_math)\n
 *       Make use of fast math operations. --use_fast_math implies --ftz=true
 *       --prec-div=false --prec-sqrt=false --fmad=true.
 *   - Preprocessing
 *     - --define-macro=<em>\<macrodef\></em> (-D)\n
 *       <em>macrodef</em> can be either <em>name</em> or
 *       <em>name=definitions</em>.
 *       - <em>name</em>\n
 *         Predefine <em>name</em> as a macro with definition 1.
 *       - <em>name=definition</em>\n
 *         The contents of <em>definition</em> are tokenized and preprocessed
 *         as if they appeared during translation phase three in a \c \#define
 *         directive.  In particular, the definition will be truncated by
 *         embedded new line characters.
 *     - --undefine-macro=<em>\<name\></em> (-U)\n
 *       Cancel any previous definition of \em name.
 *     - --include-path=<em>\<dir\></em> (-I)\n
 *       Add the directory <em>dir</em> to the list of directories to be
 *       searched for headers.  These paths are searched after the list of
 *       headers given to ::nvrtcCreateProgram.
 *     - --pre-include=<em>\<header\></em> (-include)\n
 *       Preinclude <em>header</em> during preprocessing.
 *   - Language Dialect
 *     - --std=c++11 (-std=c++11)\n
 *       Set language dialect to C++11.
 *     - --builtin-move-forward=<em>[true, false]</em> (-builtin-move-forward)\n
 *       Provide builtin definitions of std::move and std::forward, when C++11
 *       language dialect is selected.
 *      - Default : true
 *     - --builtin-initializer-list=<em>[true, false]</em> (-builtin-initializer-list)\n
 *       Provide builtin definitions of std::initializer_list class and member
 *       functions when C++11 language dialect is selected.
 *       - Default : true
 *   - Misc
 *     - --disable-warnings (-w)\n
 *       Inhibit all warning messages.
 *     - --restrict (-restrict)\n
 *       Programmer assertion that all kernel pointer parameters are restrict
 *       pointers.
 *     - --device-as-default-execution-space
*        (-default-device)\n
 *       Treat entities with no execution space annotation as \c __device__
 *       entities.
 *
 * \param   [in] prog       CUDA Online Compiler program.
 * \param   [in] numOptions Number of compiler options passed.
 * \param   [in] options    Compiler options in the form of C string array.\n
 *                          \p options can be \c NULL when \p numOptions is 0.
 *
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_OPTION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_COMPILATION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_BUILTIN_OPERATION_FAILURE \endlink
 */
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
                                int numOptions, const char **options);


/**
 * \ingroup compilation
 * \brief   ::nvrtcGetPTXSize sets \p ptxSizeRet with the size of the PTX
 *          generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * \param   [in]  prog       CUDA Online Compiler program.
 * \param   [out] ptxSizeRet Size of the generated PTX (including the trailing
 *                           \c NULL).
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetPTX
 */
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);


/**
 * \ingroup compilation
 * \brief   ::nvrtcGetPTX stores the PTX generated by the previous compilation
 *          of \p prog in the memory pointed by \p ptx.
 *
 * \param   [in]  prog CUDA Online Compiler program.
 * \param   [out] ptx  Compiled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetPTXSize
 */
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);


/**
 * \ingroup compilation
 * \brief   ::nvrtcGetProgramLogSize sets \p logSizeRet with the size of the
 *          log generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * Note that compilation log may be generated with warnings and informative
 * messages, even when the compilation of \p prog succeeds.
 *
 * \param   [in]  prog       CUDA Online Compiler program.
 * \param   [out] logSizeRet Size of the compilation log
 *                           (including the trailing \c NULL).
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetProgramLog
 */
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);


/**
 * \ingroup compilation
 * \brief   ::nvrtcGetProgramLog stores the log generated by the previous
 *          compilation of \p prog in the memory pointed by \p log.
 *
 * \param   [in]  prog CUDA Online Compiler program.
 * \param   [out] log  Compilation log.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetProgramLogSize
 */
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __NVRTC_H__ */
