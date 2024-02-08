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
#ifndef AMD_HIP_MATH_CONSTANTS_H
#define AMD_HIP_MATH_CONSTANTS_H

// single precision constants
#define HIP_INF_F            __int_as_float(0x7f800000U)
#define HIP_NAN_F            __int_as_float(0x7fffffffU)
#define HIP_MIN_DENORM_F     __int_as_float(0x00000001U)
#define HIP_MAX_NORMAL_F     __int_as_float(0x7f7fffffU)
#define HIP_NEG_ZERO_F       __int_as_float(0x80000000U)
#define HIP_ZERO_F           0.0F
#define HIP_ONE_F            1.0F
#define HIP_SQRT_HALF_F      0.707106781F
#define HIP_SQRT_HALF_HI_F   0.707106781F
#define HIP_SQRT_HALF_LO_F   1.210161749e-08F
#define HIP_SQRT_TWO_F       1.414213562F
#define HIP_THIRD_F          0.333333333F
#define HIP_PIO4_F           0.785398163F
#define HIP_PIO2_F           1.570796327F
#define HIP_3PIO4_F          2.356194490F
#define HIP_2_OVER_PI_F      0.636619772F
#define HIP_SQRT_2_OVER_PI_F 0.797884561F
#define HIP_PI_F             3.141592654F
#define HIP_L2E_F            1.442695041F
#define HIP_L2T_F            3.321928094F
#define HIP_LG2_F            0.301029996F
#define HIP_LGE_F            0.434294482F
#define HIP_LN2_F            0.693147181F
#define HIP_LNT_F            2.302585093F
#define HIP_LNPI_F           1.144729886F
#define HIP_TWO_TO_M126_F    1.175494351e-38F
#define HIP_TWO_TO_126_F     8.507059173e37F
#define HIP_NORM_HUGE_F      3.402823466e38F
#define HIP_TWO_TO_23_F      8388608.0F
#define HIP_TWO_TO_24_F      16777216.0F
#define HIP_TWO_TO_31_F      2147483648.0F
#define HIP_TWO_TO_32_F      4294967296.0F
#define HIP_REMQUO_BITS_F    3U
#define HIP_REMQUO_MASK_F    (~((~0U)<<HIP_REMQUO_BITS_F))
#define HIP_TRIG_PLOSS_F     105615.0F

// double precision constants
#define HIP_INF              __longlong_as_double(0x7ff0000000000000ULL)
#define HIP_NAN              __longlong_as_double(0xfff8000000000000ULL)
#define HIP_NEG_ZERO         __longlong_as_double(0x8000000000000000ULL)
#define HIP_MIN_DENORM       __longlong_as_double(0x0000000000000001ULL)
#define HIP_ZERO             0.0
#define HIP_ONE              1.0
#define HIP_SQRT_TWO         1.4142135623730951e+0
#define HIP_SQRT_HALF        7.0710678118654757e-1
#define HIP_SQRT_HALF_HI     7.0710678118654757e-1
#define HIP_SQRT_HALF_LO   (-4.8336466567264567e-17)
#define HIP_THIRD            3.3333333333333333e-1
#define HIP_TWOTHIRD         6.6666666666666667e-1
#define HIP_PIO4             7.8539816339744828e-1
#define HIP_PIO4_HI          7.8539816339744828e-1
#define HIP_PIO4_LO          3.0616169978683830e-17
#define HIP_PIO2             1.5707963267948966e+0
#define HIP_PIO2_HI          1.5707963267948966e+0
#define HIP_PIO2_LO          6.1232339957367660e-17
#define HIP_3PIO4            2.3561944901923448e+0
#define HIP_2_OVER_PI        6.3661977236758138e-1
#define HIP_PI               3.1415926535897931e+0
#define HIP_PI_HI            3.1415926535897931e+0
#define HIP_PI_LO            1.2246467991473532e-16
#define HIP_SQRT_2PI         2.5066282746310007e+0
#define HIP_SQRT_2PI_HI      2.5066282746310007e+0
#define HIP_SQRT_2PI_LO    (-1.8328579980459167e-16)
#define HIP_SQRT_PIO2        1.2533141373155003e+0
#define HIP_SQRT_PIO2_HI     1.2533141373155003e+0
#define HIP_SQRT_PIO2_LO   (-9.1642899902295834e-17)
#define HIP_SQRT_2OPI        7.9788456080286536e-1
#define HIP_L2E              1.4426950408889634e+0
#define HIP_L2E_HI           1.4426950408889634e+0
#define HIP_L2E_LO           2.0355273740931033e-17
#define HIP_L2T              3.3219280948873622e+0
#define HIP_LG2              3.0102999566398120e-1
#define HIP_LG2_HI           3.0102999566398120e-1
#define HIP_LG2_LO         (-2.8037281277851704e-18)
#define HIP_LGE              4.3429448190325182e-1
#define HIP_LGE_HI           4.3429448190325182e-1
#define HIP_LGE_LO           1.09831965021676510e-17
#define HIP_LN2              6.9314718055994529e-1
#define HIP_LN2_HI           6.9314718055994529e-1
#define HIP_LN2_LO           2.3190468138462996e-17
#define HIP_LNT              2.3025850929940459e+0
#define HIP_LNT_HI           2.3025850929940459e+0
#define HIP_LNT_LO         (-2.1707562233822494e-16)
#define HIP_LNPI             1.1447298858494002e+0
#define HIP_LN2_X_1024       7.0978271289338397e+2
#define HIP_LN2_X_1025       7.1047586007394398e+2
#define HIP_LN2_X_1075       7.4513321910194122e+2
#define HIP_LG2_X_1024       3.0825471555991675e+2
#define HIP_LG2_X_1075       3.2360724533877976e+2
#define HIP_TWO_TO_23        8388608.0
#define HIP_TWO_TO_52        4503599627370496.0
#define HIP_TWO_TO_53        9007199254740992.0
#define HIP_TWO_TO_54        18014398509481984.0
#define HIP_TWO_TO_M54       5.5511151231257827e-17
#define HIP_TWO_TO_M1022     2.22507385850720140e-308
#define HIP_TRIG_PLOSS       2147483648.0
#define HIP_DBL2INT_CVT      6755399441055744.0

#endif
