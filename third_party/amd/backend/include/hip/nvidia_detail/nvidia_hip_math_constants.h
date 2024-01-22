/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef NVIDIA_HIP_MATH_CONSTANTS_H
#define NVIDIA_HIP_MATH_CONSTANTS_H

#include <math_constants.h>

// single precision constants
#define HIP_INF_F            CUDART_INF_F
#define HIP_NAN_F            CUDART_NAN_F
#define HIP_MIN_DENORM_F     CUDART_MIN_DENORM_F
#define HIP_MAX_NORMAL_F     CUDART_MAX_NORMAL_F
#define HIP_NEG_ZERO_F       CUDART_NEG_ZERO_F
#define HIP_ZERO_F           CUDART_ZERO_F
#define HIP_ONE_F            CUDART_ONE_F
#define HIP_SQRT_HALF_F      CUDART_SQRT_HALF_F
#define HIP_SQRT_HALF_HI_F   CUDART_SQRT_HALF_HI_F
#define HIP_SQRT_HALF_LO_F   CUDART_SQRT_HALF_LO_F
#define HIP_SQRT_TWO_F       CUDART_SQRT_TWO_F
#define HIP_THIRD_F          CUDART_THIRD_F
#define HIP_PIO4_F           CUDART_PIO4_F
#define HIP_PIO2_F           CUDART_PIO2_F
#define HIP_3PIO4_F          CUDART_3PIO4_F
#define HIP_2_OVER_PI_F      CUDART_2_OVER_PI_F
#define HIP_SQRT_2_OVER_PI_F CUDART_SQRT_2_OVER_PI_F
#define HIP_PI_F             CUDART_PI_F
#define HIP_L2E_F            CUDART_L2E_F
#define HIP_L2T_F            CUDART_L2T_F
#define HIP_LG2_F            CUDART_LG2_F
#define HIP_LGE_F            CUDART_LGE_F
#define HIP_LN2_F            CUDART_LN2_F
#define HIP_LNT_F            CUDART_LNT_F
#define HIP_LNPI_F           CUDART_LNPI_F
#define HIP_TWO_TO_M126_F    CUDART_TWO_TO_M126_F
#define HIP_TWO_TO_126_F     CUDART_TWO_TO_126_F
#define HIP_NORM_HUGE_F      CUDART_NORM_HUGE_F
#define HIP_TWO_TO_23_F      CUDART_TWO_TO_23_F
#define HIP_TWO_TO_24_F      CUDART_TWO_TO_24_F
#define HIP_TWO_TO_31_F      CUDART_TWO_TO_31_F
#define HIP_TWO_TO_32_F      CUDART_TWO_TO_32_F
#define HIP_REMQUO_BITS_F    CUDART_REMQUO_BITS_F
#define HIP_REMQUO_MASK_F    CUDART_REMQUO_MASK_F
#define HIP_TRIG_PLOSS_F     CUDART_TRIG_PLOSS_F

// double precision constants
#define HIP_INF              CUDART_INF
#define HIP_NAN              CUDART_NAN
#define HIP_NEG_ZERO         CUDART_NEG_ZERO
#define HIP_MIN_DENORM       CUDART_MIN_DENORM
#define HIP_ZERO             CUDART_ZERO
#define HIP_ONE              CUDART_ONE
#define HIP_SQRT_TWO         CUDART_SQRT_TWO
#define HIP_SQRT_HALF        CUDART_SQRT_HALF
#define HIP_SQRT_HALF_HI     CUDART_SQRT_HALF_HI
#define HIP_SQRT_HALF_LO     CUDART_SQRT_HALF_LO
#define HIP_THIRD            CUDART_THIRD
#define HIP_TWOTHIRD         CUDART_TWOTHIRD
#define HIP_PIO4             CUDART_PIO4
#define HIP_PIO4_HI          CUDART_PIO4_HI
#define HIP_PIO4_LO          CUDART_PIO4_LO
#define HIP_PIO2             CUDART_PIO2
#define HIP_PIO2_HI          CUDART_PIO2_HI
#define HIP_PIO2_LO          CUDART_PIO2_LO
#define HIP_3PIO4            CUDART_3PIO4
#define HIP_2_OVER_PI        CUDART_2_OVER_PI
#define HIP_PI               CUDART_PI
#define HIP_PI_HI            CUDART_PI_HI
#define HIP_PI_LO            CUDART_PI_LO
#define HIP_SQRT_2PI         CUDART_SQRT_2PI
#define HIP_SQRT_2PI_HI      CUDART_SQRT_2PI_HI
#define HIP_SQRT_2PI_LO      CUDART_SQRT_2PI_LO
#define HIP_SQRT_PIO2        CUDART_SQRT_PIO2
#define HIP_SQRT_PIO2_HI     CUDART_SQRT_PIO2_HI
#define HIP_SQRT_PIO2_LO     CUDART_SQRT_PIO2_LO
#define HIP_SQRT_2OPI        CUDART_SQRT_2OPI
#define HIP_L2E              CUDART_L2E
#define HIP_L2E_HI           CUDART_L2E_HI
#define HIP_L2E_LO           CUDART_L2E_LO
#define HIP_L2T              CUDART_L2T
#define HIP_LG2              CUDART_LG2
#define HIP_LG2_HI           CUDART_LG2_HI
#define HIP_LG2_LO           CUDART_LG2_LO
#define HIP_LGE              CUDART_LGE
#define HIP_LGE_HI           CUDART_LGE_HI
#define HIP_LGE_LO           CUDART_LGE_LO
#define HIP_LN2              CUDART_LN2
#define HIP_LN2_HI           CUDART_LN2_HI
#define HIP_LN2_LO           CUDART_LN2_LO
#define HIP_LNT              CUDART_LNT
#define HIP_LNT_HI           CUDART_LNT_HI
#define HIP_LNT_LO           CUDART_LNT_LO
#define HIP_LNPI             CUDART_LNPI
#define HIP_LN2_X_1024       CUDART_LN2_X_1024
#define HIP_LN2_X_1025       CUDART_LN2_X_1025
#define HIP_LN2_X_1075       CUDART_LN2_X_1075
#define HIP_LG2_X_1024       CUDART_LG2_X_1024
#define HIP_LG2_X_1075       CUDART_LG2_X_1075
#define HIP_TWO_TO_23        CUDART_TWO_TO_23
#define HIP_TWO_TO_52        CUDART_TWO_TO_52
#define HIP_TWO_TO_53        CUDART_TWO_TO_53
#define HIP_TWO_TO_54        CUDART_TWO_TO_54
#define HIP_TWO_TO_M54       CUDART_TWO_TO_M54
#define HIP_TWO_TO_M1022     CUDART_TWO_TO_M1022
#define HIP_TRIG_PLOSS       CUDART_TRIG_PLOSS
#define HIP_DBL2INT_CVT      CUDART_DBL2INT_CVT

#endif
