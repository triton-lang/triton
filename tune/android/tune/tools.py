import isaac as sc
from numpy import mean, median
from math import ceil, exp, log, sqrt
from time import time
profile_execution_failure = (sc.OperationNotSupported,  sc.OclLaunchOutOfResources, sc.CudaLaunchOutOfResources, sc.MemObjectAllocationFailure, sc.InvalidWorkGroupSize, sc.OutOfHostMemory, sc.InvalidValue)

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
    queue.profiles[template, sc.float32] = sc.profile(template(*setting), sc.float32, queue)
    times = []
    total = 0
    i = 0
    #Warm-up
    z, events = sc.driver.enqueue(tree)
    tree.context.queues[0].synchronize()
    #Time
    while total < 1e-2:
        start = time()
        z, events = sc.driver.enqueue(tree)
        tree.context.queues[0].synchronize()
        end = time()
        times.append(end - start)
        total += times[-1]
        i+=1
    return median(times)


def tree_of(template, sizes, context):
    if issubclass(template, sc.templates.elementwise_1d):
        N, = sizes
        x = sc.empty(N, dtype=sc.float32, context=context)
        y = sc.empty(N, dtype=sc.float32, context=context)
        return sc.assign(y, x + y), (x, y)
    elif issubclass(template, sc.templates.reduce_1d):
        N, = sizes
        x = sc.empty(N, context=context)
        y = sc.empty(N, context=context)
        return sc.dot(x, y), (x, y)
    elif issubclass(template, sc.templates.elementwise_2d):
        M, N = sizes
        A = sc.empty((M,N), context=context)
        B = sc.empty((M,N), context=context)
        return A + B, (A, B)
    elif issubclass(template, sc.templates.reduce_2d):
        T = template is sc.templates.reduce_2d_cols
        M, N = sizes[::-1] if T else sizes
        A = sc.empty((M,N), context=context)
        x = sc.empty(N, context=context)
        return sc.dot(A.T, x) if T else sc.dot(A, x), (A, x)
    elif issubclass(template, sc.templates.matrix_product):
        AT = template is sc.templates.matrix_product_tn or template is sc.templates.matrix_product_tt
        BT = template is sc.templates.matrix_product_nt or template is sc.templates.matrix_product_tt
        M, N, K = sizes
        A = sc.empty((K, M) if AT else (M, K), context=context)
        B = sc.empty((N, K) if BT else (K, N), context=context)
        AA = A.T if AT else A
        BB = B.T if BT else B
        return sc.dot(AA, BB), (A, B)

def memory_footprint(template, sizes):
    if issubclass(template, sc.templates.elementwise_1d):
        return 4*3*sizes[0]*1e-9
    elif issubclass(template, sc.templates.reduce_1d):
        return 4*2*sizes[0]*1e-9
    elif issubclass(template, sc.templates.elementwise_2d):
        return 4*sizes[0]*sizes[1]*1e-9
    elif issubclass(template, sc.templates.reduce_2d):
        return 4*sizes[0]*sizes[1]*1e-9
    elif issubclass(template, sc.templates.matrix_product):
        return 4*(sizes[0]*sizes[1] + sizes[0]*sizes[2] + sizes[1]*sizes[2])*1e-9
    
def metric_of(template):
    memory_bound = [sc.templates.elementwise_1d, sc.templates.reduce_1d, sc.templates.elementwise_2d, sc.templates.reduce_2d]
    compute_bound = [sc.templates.matrix_product]
    if any([issubclass(template, x) for x in memory_bound]):
        return lambda sizes, t: memory_footprint(template, sizes)/t
    elif any([issubclass(template, x) for x in compute_bound]):
        return lambda sizes, t: 2*sizes[0]*sizes[1]*sizes[2]*1e-9/t
           
def metric_name_of(template):
    if issubclass(template, sc.templates.matrix_product):
        return 'GFLOPS'
    return 'GB/S'

def genetic_infos_of(template):
    if issubclass(template, sc.templates.elementwise_1d):
        return {'categorical': [3], 'nbits': [3,4,4,2] }
    elif issubclass(template, sc.templates.reduce_1d):
        return {'categorical': [3], 'nbits':[3,4,4,2]}
    elif issubclass(template, sc.templates.elementwise_2d):
        return {'categorical': [5], 'nbits': [3,3,3,3,4,2]}
    elif issubclass(template, sc.templates.reduce_2d):
        return {'categorical': [5], 'nbits': [3,3,3,3,4,2]}
    elif issubclass(template, sc.templates.matrix_product):
        return {'categorical': [8,9], 'nbits': [3,3,3,3,3,2,2,2,2,2,3,3]}


