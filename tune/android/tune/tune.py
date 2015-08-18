import random, argparse, json, os
from math import log, isinf
from itertools import chain, product
from numpy import argsort, argmax
from operator import mul
import isaac as sc
from isaac.external.sklearn.forest import RandomForestRegressor
import optimize, tools, model
from json import encoder
import json

encoder.FLOAT_REPR = lambda o: format(o, '.2f')
encoder.separators = (',',':')

def unique(L):
    seen = set()
    seen_add = seen.add
    return [ x for x in L if not (x in seen or seen_add(x))]

def pow2range(a, b):
    return [2**x for x in range(a, b)]


def do_tuning(device, operation, json_path): 
    #Context
    context = sc.driver.context(device)
    
    #List of size tuples to use
    sizes = {}
    sizes[sc.templates.axpy] = [(x,) for x in tools.expspace(1e3, 1e8, 4)]
    sizes[sc.templates.gemv_n] = product(pow2range(4,17), pow2range(4,17))
    sizes[sc.templates.gemv_t] = sizes[sc.templates.gemv_n]
    sizes[sc.templates.gemm_nn]     = product(pow2range(5, 12), pow2range(5, 12), pow2range(5, 15))
    sizes[sc.templates.gemm_tn]     = sizes[sc.templates.gemm_nn]
    sizes[sc.templates.gemm_nt]     = sizes[sc.templates.gemm_nn]
    sizes[sc.templates.gemm_tt]     = sizes[sc.templates.gemm_nn]
    

    #Quick tuning - AlexNet sizes + Intuition
    quick_tuning = False
    if quick_tuning:
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
    profiles, X, Y = [], [], []
    
	#Restore previous run
    import csv
    savepath = os.path.join('save', operation.__name__)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
	
    try:
		with open(os.path.join(savepath, 'X.csv')) as f:
			X = [tuple(map(int, row)) for row in csv.reader(f, delimiter=',')]
			
		with open(os.path.join(savepath, 'Y.csv')) as f:
			Y = [map(float, row) for row in csv.reader(f, delimiter=',')]
		
		with open(os.path.join(savepath, 'profiles.csv')) as f:
			def mmap(x):
				if x=='FETCH_FROM_LOCAL':
					return sc.templates.fetching_policy_type.FETCH_FROM_LOCAL
				if x=='FETCH_FROM_GLOBAL_CONTIGUOUS':
					return sc.templates.fetching_policy_type.FETCH_FROM_GLOBAL_CONTIGUOUS
				if x=='FETCH_FROM_GLOBAL_STRIDED':
					return sc.templates.fetching_policy_type.FETCH_FROM_GLOBAL_STRIDED
				return int(x)
			profiles = [map(mmap,row) for v in row for row in csv.reader(f, delimiter=',')]
    except:
		raise
	
    for idx, x in enumerate(sizes):
        if x in X:
			continue
        print x
        idx = len(X)
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
                clf = RandomForestRegressor(min(10, idx+1), max_depth=min(10, idx+1)).fit(X, Y)
                #clf, nrmse = model.train(X, Y, profiles)
                predperf = clf.predict(x)[0]
                best = (-predperf).argsort()[:5]
                perf = []
                for b in best:
                    try:
                        perf += [performance(x, tools.benchmark(operation, profiles[b], tree))]
                    except (sc.OperationNotSupported, sc.LaunchOutOfResources, sc.MemObjectAllocationFailure):
                        pass
                predicted = profiles[best[argmax(perf)]]
            tune = not optimize.is_local_optimum(predicted, operation, x, context)     
            #tune = True
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
        
        for (fname, data) in zip(['X.csv', 'Y.csv', 'profiles.csv'], [X, Y, profiles]):
            with open(os.path.join(savepath, fname), 'wb') as f:
                csv.writer(f).writerows(data)

    
    #Export to JSON
    json_path = tools.sanitize(device.name) + '.json' if not json_path else json_path
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
        clf, nrmse = model.train(X, Y, profiles)
        D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                            'children_right': e.tree_.children_right.tolist(),
                            'threshold': e.tree_.threshold.astype('float64').tolist(),
                            'feature': e.tree_.feature.astype('float64').tolist(),
                            'value': e.tree_.value[:,:,0].astype('float64').tolist()} for e in clf.estimators_]
    D['profiles'] = [map(int, x) for x in profiles]
    json.dump(json_data, open(json_path,'w'))

