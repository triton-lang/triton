# Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
# 
# This file is part of ISAAC.
# 
# ISAAC is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301  USA

import isaac as sc
from numpy import mean, median
from math import ceil, exp, log, sqrt
import time
profile_execution_failure = (sc.OperationNotSupported,  sc.OclLaunchOutOfResources, sc.CudaLaunchOutOfResources, sc.MemObjectAllocationFailure, sc.InvalidWorkGroupSize, sc.OutOfHostMemory, sc.InvalidValue)

dtype=sc.float32

def sanitize(string, keep_chars = ['_']):
    string = string.replace(' ', '_').replace('-', '_').lower()
    string = "".join(c for c in string if c.isalnum() or c in keep_chars).rstrip()
    return string
    
def distance(x, y):
    return sqrt(sum([(a - b)**2 for a, b in zip(x, y)]))

def linspace(a, b, n=100):
    if n < 2:
        return b
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a  for i in range(n)]
    
def expspace(a,b,N,r=128):
    return [int(ceil(exp(x)/r)*r) for x in linspace(log(a), log(b), N)]


def benchmark(template, tree, operation=sc.templates.gemm_nn):
    queue = tree.context.queues[0]
    queue.profiles[template, dtype] = sc.profile(template, dtype, queue)
    times = []
    total = 0
    #Warm-up
    try:
        z, events = sc.driver.enqueue(tree)
        queue.synchronize()
    except:
        return float("inf")
    #Time
    while total < 1e-2:
        start = time.time()
        z, events = sc.driver.enqueue(tree)
        queue.synchronize()
        end = time.time()
        times.append(end - start)
        total += times[-1]
    return median(times)


def tree_of(template, sizes, context):
    if issubclass(template, sc.templates.elementwise_1d):
        N, = sizes
        x = sc.empty(N, dtype=dtype, context=context)
        y = sc.empty(N, dtype=dtype, context=context)
        return sc.assign(y, x + y), (x, y)
    elif issubclass(template, sc.templates.reduce_1d):
        N, = sizes
        x = sc.empty(N, dtype=dtype, context=context)
        y = sc.empty(N, dtype=dtype, context=context)
        return sc.dot(x, y), (x, y)
    elif issubclass(template, sc.templates.elementwise_2d):
        M, N = sizes
        A = sc.empty((M,N), dtype=dtype, context=context)
        B = sc.empty((M,N), dtype=dtype, context=context)
        return A + B, (A, B)
    elif issubclass(template, sc.templates.reduce_2d):
        T = template is sc.templates.reduce_2d_cols
        M, N = sizes[::-1] if T else sizes
        A = sc.empty((M,N), dtype=dtype, context=context)
        x = sc.empty(N, dtype=dtype, context=context)
        y = sc.empty(M, dtype=dtype, context=context)
        return sc.assign(x, sc.dot(A.T, y)) if T else sc.assign(y, sc.dot(A, x)), (A, x, y)
    elif issubclass(template, sc.templates.gemm):
        AT = template is sc.templates.gemm_tn or template is sc.templates.gemm_tt
        BT = template is sc.templates.gemm_nt or template is sc.templates.gemm_tt
        M, N, K = sizes
        C = sc.empty((M,N), dtype=dtype, context=context)
        A = sc.empty((K, M) if AT else (M, K), dtype=dtype, context=context)
        B = sc.empty((N, K) if BT else (K, N), dtype=dtype, context=context)
        AA = A.T if AT else A
        BB = B.T if BT else B
        return sc.assign(C, sc.dot(AA, BB)), (A, B, C)

def memory_footprint(template, sizes):
    if issubclass(template, sc.templates.elementwise_1d):
        return 4*3*sizes[0]*1e-9
    elif issubclass(template, sc.templates.reduce_1d):
        return 4*2*sizes[0]*1e-9
    elif issubclass(template, sc.templates.elementwise_2d):
        return 4*sizes[0]*sizes[1]*1e-9
    elif issubclass(template, sc.templates.reduce_2d):
        return 4*sizes[0]*sizes[1]*1e-9
    elif issubclass(template, sc.templates.gemm):
        return 4*(sizes[0]*sizes[1] + sizes[0]*sizes[2] + sizes[1]*sizes[2])*1e-9
    
def metric_of(template):
    memory_bound = [sc.templates.elementwise_1d, sc.templates.reduce_1d, sc.templates.elementwise_2d, sc.templates.reduce_2d]
    compute_bound = [sc.templates.gemm]
    if any([issubclass(template, x) for x in memory_bound]):
        return lambda sizes, t: memory_footprint(template, sizes)/t
    elif any([issubclass(template, x) for x in compute_bound]):
        return lambda sizes, t: 2*sizes[0]*sizes[1]*sizes[2]*1e-9/t
           
def metric_name_of(template):
    if issubclass(template, sc.templates.gemm):
        return 'GFLOPS'
    return 'GB/S'

def external_profiles(template):
    res = []
    if template is sc.templates.gemm_nn:
        res += [sc.templates.cublas_gemm('N','N')]
        res += [sc.templates.intelblas_gemm('N','N')]
        res += [sc.templates.intelblas_gemm_image('N','N')]
    elif template is sc.templates.gemm_tn:
        res += [sc.templates.cublas_gemm('T','N')]
        res += [sc.templates.intelblas_gemm('T','N')]
        res += [sc.templates.intelblas_gemm_image('T','N')]
    elif template is sc.templates.gemm_nt:
        res += [sc.templates.cublas_gemm('N','T')]
        res += [sc.templates.intelblas_gemm('N','T')]
        res += [sc.templates.intelblas_gemm_image('N','T')]
    elif template is sc.templates.gemm_tt:
        res += [sc.templates.cublas_gemm('T','T')]
        res += [sc.templates.intelblas_gemm('T','T')]
        res += [sc.templates.intelblas_gemm_image('T','T')]
    elif template is sc.templates.reduce_2d_cols:
        res += [sc.templates.intelblas_gemv()]
    elif template is sc.templates.reduce_2d_rows:
        res += [sc.templates.intelblas_gemv()]
    return res
        
def genetic_infos_of(template):
    if issubclass(template, sc.templates.elementwise_1d):
        return {'categorical': [], 'nbits': [3,4,4] }
    elif issubclass(template, sc.templates.reduce_1d):
        return {'categorical': [], 'nbits':[3,4,4]}
    elif issubclass(template, sc.templates.elementwise_2d):
        return {'categorical': [], 'nbits': [3,3,3,3,4]}
    elif issubclass(template, sc.templates.reduce_2d):
        return {'categorical': [], 'nbits': [3,3,3,3,4]}
    elif issubclass(template, sc.templates.gemm):
        return {'categorical': [], 'nbits': [3,3,3,3,3,2,2,2,3,3]}

def convert(profile):
	if isinstance(profile, str):
		return profile
	else:
		return map(int, profile)
