from __future__ import print_function
import argparse
#External modules
import isaac as sc
import numpy as np
#Tuner modules
import dataset
from export import export
import regression
import operations as op

def parse_arguments():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default=0, type=int, help='Device to tune for')
    parser.add_argument('--database', default=0, type=str, help='Location of the database to update')
    parser.add_argument('--conv', default = False, action='store_true', help='Tune CONV')
    parser.add_argument('--gemm', default = False, action='store_true', help='Tune GEMM')
    parser.add_argument('--pool', default = False, action='store_true', help='Tune POOL')
    parser.add_argument('--nsamples', default=20000, type=int, help='Number of training samples to generate')
    args = parser.parse_args()

    # Operations
    ops, wraps = ['conv','gemm','pool'], [sc.templates.Conv, sc.templates.GEMM, sc.templates.Pool]
    ops = [wrap for operation, wrap in zip(ops, wraps) if getattr(args, operation)]

    # Done
    return (args.database, args.device, ops, args.nsamples)

def cuda_environment(device):
    platforms = sc.driver.platforms()
    devices = [d for platform in platforms for d in platform.devices]
    device = devices[device]
    context = sc.driver.Context(device)
    stream = sc.driver.Stream(context)
    return device, context, stream
    
if __name__ == "__main__":
    # Get arguments
    database, device, operations, nsamples = parse_arguments()
    
    # Initialize CUDA environment
    init_cuda = lambda: cuda_environment(device)
    
    # Run the auto-tuning
    for OpType in operations:
        print("----------------")
        print('Now tuning {}:'.format(OpType.id))
        print("----------------")
        X, Y = dataset.benchmarks(OpType, nsamples, init_cuda)
        model = regression.train(OpType, X, Y)
        kernels = regression.prune(OpType, model, init_cuda)
        export(database, kernels, model, OpType.id, init_cuda)
