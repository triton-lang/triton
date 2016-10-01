import isaac as sc
import isaac.templates as templates

sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE
sc.driver.default.device = 0;

#Construct vectors using the default device.
M, N, K = 972, 1541, 793
A = sc.empty((M, K), sc.float32)
B = sc.empty((K, N), sc.float32)

#Get command queue
queue = A.context.queues[0]

#Benchmark profile 1
queue.profiles[sc.templates.gemm_nn, sc.float32] = sc.profile(templates.gemm_nn(1,8,16,8,1,8,1,8,templates.FETCH_FROM_LOCAL,templates.FETCH_FROM_LOCAL,8,8), sc.float32, queue)
C, events = sc.driver.enqueue(sc.reduce_1d(A, B))
C.context.synchronize()
print 'Profile 1 finished in', sum([e.elapsed_time for e in events])*1e-9, 's'

#Benchmark profile 2
queue.profiles[sc.templates.gemm_nn, sc.float32] = sc.profile(templates.gemm_nn(1,8,16,16,1,8,1,8,templates.FETCH_FROM_LOCAL,templates.FETCH_FROM_LOCAL,8,16), sc.float32, queue)
C, events = sc.driver.enqueue(sc.reduce_1d(A, B))
C.context.synchronize()
print 'Profile 2 finished in', sum([e.elapsed_time for e in events])*1e-9, 's'
