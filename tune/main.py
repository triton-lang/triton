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
    parser.add_argument("-j", "--json", default='', type=str)
    parser.add_argument('--elementwise_1d', action='store_true', help='Tune ELEMENTWISE [1D]')
    parser.add_argument('--elementwise_2d', action='store_true', help='Tune ELEMENTWISE [2D]')
    parser.add_argument('--reduce_1d', action='store_true', help='Tune REDUCE [1D]')
    parser.add_argument('--reduce_2d_rows', action='store_true', help='Tune REDUCE [2D/rows]')
    parser.add_argument('--reduce_2d_cols', action='store_true', help='Tune REDUCE [2D/cols]')
    parser.add_argument('--matrix_product_nn', action='store_true', help='Tune MATRIX PRODUCT [NN]')
    parser.add_argument('--matrix_product_tn', action='store_true', help='Tune MATRIX PRODUCT [TN]')
    parser.add_argument('--matrix_product_nt', action='store_true', help='Tune MATRIX PRODUCT [NT]')
    parser.add_argument('--matrix_product_tt', action='store_true', help='Tune MATRIX PRODUCT [TT]')
    args = parser.parse_args()
    
    #Device
    device = devices[int(args.device)]
    print("----------------")
    print("Devices available:")
    print("----------------")
    for (i, d) in enumerate(devices):
        selected = '[' + ('x' if device==d else ' ') + ']'
        print selected , '-',  sc.driver.device_type_to_string(d.type), '-', d.name, 'on', d.platform.name
    
    #Operations
    operations = ['elementwise_1d', 'reduce_1d', 'elementwise_2d', 'reduce_2d_rows', 'reduce_2d_cols', 'matrix_product_nn', 'matrix_product_tn', 'matrix_product_nt', 'matrix_product_tt']
    operations = [getattr(sc.templates,op) for op in operations  if getattr(args, op)]
        
    return (device, operations, args.json)
        

class ProgressBar:
    
    def __init__(self, length,  metric_name):
        self.length = length
        self.metric_name = metric_name
    
    def set_prefix(self, prefix):
        self.prefix = prefix
        sys.stdout.write("{0}: [{1}] {2: >3}%".format(prefix.ljust(17), ' '*self.length, 0))
        sys.stdout.flush()
        
    def set_finished(self):
        sys.stdout.write("\n")
        
    def update(self, i, total, x, y, complete=False):
        percent = float(i) / total
        hashes = '#' * int(round(percent * self.length))
        spaces = ' ' * (self.length - len(hashes))
        #Format of structures to print
        xformat = ','.join(map(str,map(int, x)))
        yformat = int(y)
        percentformat = int(round(percent * 100))
        sys.stdout.write(("\r{0}: [{1}] {2: >3}% [{3} {4}] ({5})").format(self.prefix.ljust(17), hashes + spaces, percentformat, yformat, self.metric_name, xformat))
        sys.stdout.flush()
        
if __name__ == "__main__":    
    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    sc.driver.default.queue_properties = sc.driver.PROFILING_ENABLE
    device, operations, json = parse_arguments()
    
    for operation in operations:
        tuner = Tuner(logger, device, operation, json, ProgressBar(30, metric_name_of(operation)))
        tuner.run(level='intermediate')
