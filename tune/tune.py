import random, argparse, json, os
from math import log, isinf
from itertools import chain, product
from numpy import argsort, argmax
from operator import mul
from sklearn import ensemble
import isaac as sc
import optimize, tools, model

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
encoder.separators = (',',':')

def unique(L):
    seen = set()
    seen_add = seen.add
    return [ x for x in L if not (x in seen or seen_add(x))]

def pow2range(a, b):
    return [2**x for x in range(a, b)]


def tune(device, operation, json_path): 
    #List devices
    platforms = sc.driver.get_platforms()
    context = sc.driver.context(device)
    
    #List of size tuples to use
    sizes = {}
    sizes[sc.templates.axpy] = [(x,) for x in tools.expspace(1e3, 1e8, 4)]
    sizes[sc.templates.gemv_n] = product(pow2range(4,17), pow2range(4,17))
    sizes[sc.templates.gemv_t] = sizes[sc.templates.gemv_n]
    sizes[sc.templates.gemm_nn]     = product(pow2range(6, 12), pow2range(6, 12), pow2range(6, 12))
    sizes[sc.templates.gemm_tn]     = sizes[sc.templates.gemm_nn]
    sizes[sc.templates.gemm_nt]     = sizes[sc.templates.gemm_nn]
    sizes[sc.templates.gemm_tt]     = sizes[sc.templates.gemm_nn]
    

    #Quick tuning - AlexNet sizes + Intuition
    sizes[sc.templates.ger] 		 = [(1536,1536)]

    sizes[sc.templates.gemv_n]		 = [(1000,256),
                                        (4096,256)]
    sizes[sc.templates.gemv_t]		 = [(169,256),
                                        (169,384),
                                        (729,256),
                                        (3025,96)]
	
    sizes[sc.templates.gemm_nn]	 = [(3025,96,363),
                                        (729,128,1200),
                                        (169,384,2304),
                                        (169,192,1728),
                                        (169,128,1728)]
    sizes[sc.templates.gemm_nt]	 = [(169,1728,128),
										(169,1728,192),
										(169,2304,384),
										(729,1200,128)]
    sizes[sc.templates.gemm_tn]	 = [(1728,128,169), 
										(1728,192,169),
										(2304,384,169),
										(1200,128,729),
										(363,96,3025)]
    
    #Remove duplicated
    sizes = unique(list(sizes[operation]))
    sizes = [x for x in sizes if 1e-4 <= tools.memory_footprint(operation, x) <= 1e-1]

    #Training data
    performance = tools.metric_of(operation)
    profiles = []
    X = []
    Y = []
    for idx, x in enumerate(sizes):
        print x
        nparams = len(profiles)
        tree, operands = tools.tree_of(operation, x, context)
        #Check if the current best prediction is not a local optimum
        if idx==0:
            tune = True
            predicted = None
        else:
            if nparams==1:
                predicted = profiles[0]
            else:
                clf = ensemble.RandomForestRegressor(min(10, idx+1), max_depth=min(10, idx+1)).fit(X, Y)
                #clf, nrmse = profile.train(X, Y, profiles)
                predperf = clf.predict(x)[0]
                best = (-predperf).argsort()[:5]
                perf = [performance(x, tools.benchmark(operation, profiles[b], tree)) for b in best]
                predicted = profiles[best[argmax(perf)]]
            #tune = not optimize.is_local_optimum(predicted, operation, x, context)     
            tune = True
        #Retune if necessary
        if tune:
            #new = optimize.exhaustive(operation, x, context)
            new = optimize.genetic(operation, x, context, niter=1000, naccept=1000, popsize=20, prior=predicted)[0]
            if new not in profiles:
                profiles.append(new)
                if idx > 0:
                    for xx,yy in zip(X, Y):
                        _tree, _operands = tools.tree_of(operation, xx, context)
                        try:
                            time = tools.benchmark(operation, new, _tree)
                            perf = performance(xx, time)
                        except (sc.OperationNotSupported, sc.LaunchOutOfResources, sc.MemObjectAllocationFailure):
                            perf = 0
                        yy.append(0 if isinf(perf) else perf)
        #Update dataset
        y = []
        fastest = max(predperf) if nparams > 1 else None
        for ip, p in enumerate(profiles):
            try:
                perf = 0 if fastest and ip < nparams and predperf[ip]/fastest < .1 else performance(x,tools.benchmark(operation, p, tree))
            except (sc.OperationNotSupported, sc.LaunchOutOfResources, sc.MemObjectAllocationFailure):
                perf = 0
            y.append(0 if isinf(perf) else perf)
        X.append(x)
        Y.append(y)

    
    #Export to JSON
    if os.path.isfile(json_path):
        json_data = json.load(open(json_path, 'r'))
    else:
        json_data = {}
        json_data["version"] = "1.0"
    operation_name = operation.__name__
    if operation_name not in json_data:
        json_data[operation_name] = {}
    json_data[operation_name]['float32'] = {}
    D = json_data[operation_name]['float32']
    if len(profiles) > 1:
        clf, nrmse = profile.train(X, Y, profiles)
        D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                            'children_right': e.tree_.children_right.tolist(),
                            'threshold': e.tree_.threshold.astype('float64').tolist(),
                            'feature': e.tree_.feature.astype('float64').tolist(),
                            'value': e.tree_.value[:,:,0].astype('float64').tolist()} for e in clf.estimators_]
    D['profiles'] = [map(int, x) for x in profiles]
    json.dump(json_data, open(json_path,'w'))
    

def parse_arguments():
    platforms = sc.driver.get_platforms()
    devices = [d for platform in platforms for d in platform.get_devices()]
    #Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default=0, type=int, help='Device to tune for')
    parser.add_argument("-o", "--operation", type=str, required=True, help='Operation to tune for')
    parser.add_argument("-j", "--json", default='', type=str)
    args = parser.parse_args()
    
    device = devices[int(args.device)]
    print("----------------")
    print("Devices available:")
    print("----------------")
    for (i, d) in enumerate(devices):
        selected = '[' + ('x' if device==d else ' ') + ']'
        print selected , '-',  sc.driver.device_type_to_string(d.type), '-', d.name, 'on', d.platform.name
    print("----------------")
    
    
    operation = {'axpy': sc.templates.axpy, 'dot': sc.templates.dot,
                 'ger': sc.templates.ger, 'gemv_n': sc.templates.gemv_n, 'gemv_t': sc.templates.gemv_t,
                 'gemm_nn': sc.templates.gemm_nn, 'gemm_tn': sc.templates.gemm_tn, 'gemm_nt': sc.templates.gemm_nt, 'gemm_tt':sc.templates.gemm_tt}[args.operation]
    json = tools.sanitize(device.name) + '.json' if not args.json else args.json
        
    return (device, operation, json)
        
            
if __name__ == "__main__":
    sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE
    args = parse_arguments()
    tune(*args)

