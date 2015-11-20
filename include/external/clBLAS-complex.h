/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#ifndef CLBLAS_COMPLEX_H_
#define CLBLAS_COMPLEX H_

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_float2 FloatComplex;
typedef cl_double2 DoubleComplex;

static __inline FloatComplex
floatComplex(float real, float imag)
{
    FloatComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

static __inline DoubleComplex
doubleComplex(double real, double imag)
{
    DoubleComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

#define CREAL(v) ((v).s[0])
#define CIMAG(v) ((v).s[1])

#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif /* CLBLAS_COMPLEX_H_ */
