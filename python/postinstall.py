import sys

def add_input(help, default):
    sys.stdout.write(help + "[" + default + "] : ")
    sys.stdout.flush()
    return raw_input() or default

retune = add_input("No profile was found. Do you want to create one? (y/n)", 'y')
if retune=='y':
    import subprocess
    #Interactive
    opts = []
    subprocess.call(["${CMAKE_BINARY_DIR}/python/autotune/dist/autotune","list-devices"])
    opts += ['--device'] + [add_input('Device to tune for','0')]
    print '----------------'
    opts += ['--operations'] + [add_input('Operations to tune for','vaxpy-float32,maxpy-float32,dot-float32,gemv-float32,gemm-float32')]
    print '----------------'
    opts += ['--gemv-layouts'] + [add_input('GEMV Layouts', 'N,T')]
    print '----------------'
    opts += ['--gemm-layouts'] + [add_input('GEMM Layouts', 'NN,NT,TN,TT')]
    print '----------------'
    print 'Methods:'
    print 'simple: Tune the operations for a specific size.'
    print 'full: Build input-dependent models for the operation (Better performance, but can take up to a day)'
    opts += [add_input("Method:", 'simple')]
    if opts[-1] == 'simple':
        print '----------------'
        opts += ['--blas1-size'] + [add_input('BLAS1 size', '10e6')]
        print '----------------'
        opts += ['--blas2-size'] + [add_input('BLAS2 sizes (M,N)', '2560,2560').split(',')]
        print '----------------'
        opts += ['--blas3-size'] + [add_input('BLAS3 sizes (M,N,K)', '1024,1024,1024').split(',')]
    print '----------------'
    subprocess.call(["${CMAKE_BINARY_DIR}/python/autotune/dist/autotune", "tune"] + opts +['--json', 'tmp.json'])
    

