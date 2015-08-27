import argparse, logging, sys
import isaac as sc
from tune.tune import Tuner
from tune.tools import metric_name_of

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
        

class ProgressBar:
    
    def __init__(self, length,  metric_name):
        self.length = length
        self.metric_name = metric_name
    
    def set_prefix(self, prefix):
        self.prefix = prefix
        
    def update(self, i, total, performance):
        percent = float(i) / total
        hashes = '#' * int(round(percent * self.length))
        spaces = ' ' * (self.length - len(hashes))
        sys.stdout.write(("\r" + self.prefix.ljust(10) + ": [{0}] {1: >3}% [{2} " + self.metric_name + "]").format(hashes + spaces, int(round(percent * 100)), int(performance)))
        sys.stdout.flush()
        
if __name__ == "__main__":    
    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE
    device, operation, json = parse_arguments()
    tuner = Tuner(logger, device, operation, json, ProgressBar(30, metric_name_of(operation)))
    tuner.run()
