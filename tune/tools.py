import isaac as isc
from numpy import mean, median
from math import ceil, exp, log, sqrt

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
                  
def benchmark(template, setting, tree):
    queue = tree.context.queues[0]
    queue.profiles[template, isc.float32] = isc.profile(template(*setting), isc.float32, queue)
    times = []
    total = 0
    i = 0
    while total < 1e-2:
        #z = isc.zeros(1, 10000000, isc.float32, tree.context)
        z, events = isc.driver.enqueue(tree)
        tree.context.queues[0].synchronize()
        times.append(1e-9*sum([e.elapsed_time for e in events]))
        total += times[-1]
        i+=1
    return mean(times)


def tree_of(template, sizes, context):
    if issubclass(template, isc.templates.axpy):
        N, = sizes
        x = isc.empty(N, dtype=isc.float32, context=context)
        y = isc.empty(N, dtype=isc.float32, context=context)
        return x + y, (x, y)
    elif issubclass(template, isc.templates.dot):
        N, = sizes
        x = isc.empty(N, context=context)
        y = isc.empty(N, context=context)
        return isc.dot(x, y), (x, y)
    elif issubclass(template, isc.templates.ger):
        M, N = sizes
        A = isc.empty((M,N), context=context)
        B = isc.empty((M,N), context=context)
        return A + B, (A, B)
    elif issubclass(template, isc.templates.gemv):
        T = template is isc.templates.gemv_t
        M, N = sizes[::-1] if T else sizes
        A = isc.empty((M,N), context=context)
        x = isc.empty(N, context=context)
        return isc.dot(A.T, x) if T else isc.dot(A, x), (A, x)
    elif issubclass(template, isc.templates.gemm):
        AT = template is isc.templates.gemm_tn or template is isc.templates.gemm_tt
        BT = template is isc.templates.gemm_nt or template is isc.templates.gemm_tt
        M, N, K = sizes
        A = isc.empty((K, M) if AT else (M, K), context=context)
        B = isc.empty((N, K) if BT else (K, N), context=context)
        AA = A.T if AT else A
        BB = B.T if BT else B
        return isc.dot(AA, BB), (A, B)

def memory_footprint(template, sizes):
    if issubclass(template, isc.templates.axpy):
        return 4*3*sizes[0]*1e-9
    elif issubclass(template, isc.templates.dot):
        return 4*2*sizes[0]*1e-9
    elif issubclass(template, isc.templates.ger):
        return 4*3*sizes[0]*sizes[1]*1e-9
    elif issubclass(template, isc.templates.gemv):
        return 4*sizes[0]*sizes[1]*1e-9
    elif issubclass(template, isc.templates.gemm):
        return 4*(sizes[0]*sizes[1] + sizes[0]*sizes[2] + sizes[1]*sizes[2])*1e-9
    
def metric_of(template):
    memory_bound = [isc.templates.axpy, isc.templates.dot, isc.templates.ger, isc.templates.gemv]
    compute_bound = [isc.templates.gemm]
    if any([issubclass(template, x) for x in memory_bound]):
        return lambda sizes, t: memory_footprint(template, sizes)/t
    elif any([issubclass(template, x) for x in compute_bound]):
        return lambda sizes, t: 2*sizes[0]*sizes[1]*sizes[2]*1e-9/t
                
def genetic_infos_of(template):
    if issubclass(template, isc.templates.axpy):
        return {'categorical': [3], 'nbits': [3,4,4,2] }
    elif issubclass(template, isc.templates.dot):
        return {'categorical': [3], 'nbits':[3,4,4,2]}
    elif issubclass(template, isc.templates.ger):
        return {'categorical': [5], 'nbits': [3,3,3,3,4,2]}
    elif issubclass(template, isc.templates.gemv):
        return {'categorical': [5], 'nbits': [3,3,3,3,4,2]}
    elif issubclass(template, isc.templates.gemm):
        return {'categorical': [8,9], 'nbits': [3,3,3,3,3,2,2,2,2,2,3,3]}


