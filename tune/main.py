import argparse
import isaac as sc
from isaac.autotuning.tune import tune

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
    
    return (device, operation, args.json)
        
            
if __name__ == "__main__":
    sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE
    args = parse_arguments()
    tune(*args)
