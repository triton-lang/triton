import isaac as sc
FETCH_FROM_LOCAL = sc.templates.fetching_policy_type.FETCH_FROM_LOCAL
#Enable profiling info
sc.state.queue_properties = sc.CL_QUEUE_PROFILING_ENABLE

context = sc.context(sc.get_platforms()[0].get_devices()[0])
#Construct vectors using the default device.
M, N, K = 972, 1541, 793
A = sc.empty((M, K), sc.float32, context)
B = sc.empty((K, N), sc.float32, context)
#Get command queue
queue = A.context.queues[0]
#Benchmark profile 1
queue.models[sc.templates.gemm_nn, sc.float32] = sc.model(sc.float32, sc.templates.gemm_nn(1,8,16,8,1,8,1,8,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8), queue)
C, events = sc.enqueue(sc.dot(A, B))
C.context.queues[0].synchronize()
print 'Profile 1 finished in', sum([e.elapsed_time for e in events])*1e-9, 's'
#Benchmark profile 2
queue.models[sc.templates.gemm_nn, sc.float32] = sc.model(sc.float32, sc.templates.gemm_nn(1,8,16,16,1,8,1,8,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,16), queue)
C, events = sc.enqueue(sc.dot(A, B))
C.context.queues[0].synchronize()
print 'Profile 2 finished in', sum([e.elapsed_time for e in events])*1e-9, 's'
