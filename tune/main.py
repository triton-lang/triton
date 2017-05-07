import argparse
#External modules
import isaac as sc
import numpy as np
#Tuner modules
import dataset
from export import export
import regression
import operations as op
from tools import mkdir

def parse_arguments():
    platforms = sc.driver.platforms()
    devices = [d for platform in platforms for d in platform.devices]
    
    #Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default=0, type=int, help='Device to tune for')
    parser.add_argument('--database', default=0, type=str, help='Location of the database to update')
    parser.add_argument('--conv', action='store_true', help='Tune CONV')
    parser.add_argument('--gemm', default = False, action='store_true', help='Tune GEMM')
    parser.add_argument('--nsamples', default=20000, type=int, help='Number of training samples to generate')
    args = parser.parse_args()
    
    #Device
    device = devices[int(args.device)]
    print("----------------")
    print("Devices available:")
    print("----------------")
    for (i, d) in enumerate(devices):
        selected = '[' + ('x' if device==d else ' ') + ']'
        print(selected , '-', d.name, 'on', d.platform.name)
    
    #Operations
    ops, wraps = ['conv','gemm'], [op.ConvWrapper, op.GEMMWrapper]
    operations = [wrap for op, wrap in zip(ops, wraps) if getattr(args, op)]

    
    return (args.database, device, operations, args.nsamples)

def gmean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))

if __name__ == "__main__":    
    database, device, operations, nsamples = parse_arguments()
    for OpType in operations:
        opname = OpType.name
        arch = 'sm_' + '_'.join(map(str, device.compute_capability))
        prefix = 'save/{}/{}'.format(opname, arch)
        mkdir(prefix)
        print('===============')
        print('Now tuning {}:'.format(opname))
        print('===============')
        X, Y = dataset.benchmarks(prefix, OpType, device, nsamples)
        model = regression.train(prefix, OpType, X, Y)
        kernels = regression.prune(prefix, OpType, device, model)
        export(database, device, kernels, model, opname)
        speedups = []
        for shape in OpType.bench_shapes():
            cufn = op.cudaGemm if OpType == op.GEMMWrapper else op.cudaConv
            cuperf = cufn(sc.driver.default_context(), sc.driver.default_stream(), *shape)
            y, _ = regression.maximize(OpType, device, model, shape, kernels)
            speedups += [y/cuperf]
            print(shape, y, cuperf)
        print(gmean(speedups))
        
        
