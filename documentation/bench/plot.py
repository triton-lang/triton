import matplotlib.pyplot as plt
import numpy as np

def add_line(ax, xpos, ypos, height=.1):
    line = plt.Line2D([xpos, xpos], [ypos + height, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)
    
bench = [('DeepBench-Forward\nM=K=1760', 'N'),
         ('DeepBench-Backward\nM=K=2560', 'N'),
         ('Covariance\nK=60000', 'M=N'),
         ('Blocked SVD\nK=32', 'M=N')]

labels = [[16, 32, 64, 128, 7000],
         [16, 32, 64, 128, 7000],
         [32, 256],
         [896, 3456, 4096]]

configs = {
  'Pascal Titan X': {'lib': 'cuBLAS',
                     'libperf': [1.65, 1.88, 2.58, 4.83, 11.5,
                                 0.72, 1.72, 2.39, 2.86, 7.77,
                                 0.80, 3.61,
                                 1.37, 2.50, 2.57],
                     'libcol': 'green',
                     'scperf': [1.15, 2.43, 3.83, 5.53, 11.5,
                              1.78, 3.06, 4.37, 5.52, 8.67,
                              1.44, 6.43,
                              1.14, 4.53, 4.91]},

  'R9 Fury': {'lib': 'clBLAS',
              'libperf': [0.22, 0.65, 1.35, 1.92, 3.35,
                          0.28, 0.64, 1.36, 1.91, 3.32,
                          0.02, 0.87,
                          0.43, 0.98, 1.95],
              'libcol': '#d30034',
              'scperf':  [0.67, 0.94, 1.18, 2.12, 4.66,
                          0.63, 1.15, 1.43, 1.82, 4.22,
                          0.19, 2.82,
                          0.35, 1.82, 1.80]}
}

for device, conf in configs.iteritems():
    width = 0.5
    sep = 1.3
    xx = sep*np.arange(len(conf['scperf'])) + width
    groups = [0] + [len(_) for _ in labels]
    for i in np.cumsum(groups)[:-1]:
        xx[i:] += sep
    xmax = xx[-1] + width + sep
    figure, ax = plt.subplots(figsize=(12,8))
    sc = ax.bar(xx - width, conf['scperf'], width, color='purple')
    cu = ax.bar(xx, conf['libperf'], width, color=conf['libcol'])
    linex = [(xx[i] - sep) for i in np.cumsum(groups)[1:-1]]
    linex = [0] + linex + [xmax]
    for i in range(len(linex)-1):
        group, sublabel = bench[i]
        add_line(ax, linex[i]/xmax, 0, -10)
        ax.text(.5*(linex[i] + linex[i+1])/xmax, -.12, group, ha='center', transform=ax.transAxes, fontsize = 10, color='darkblue')
        ax.text(.5*(linex[i] + linex[i+1])/xmax, -.07, sublabel, ha='center', transform=ax.transAxes, fontsize = 10)
    ax.set_xlim((0,xmax))
    ax.set_xticks(xx)
    ax.set_xticklabels([x for _ in labels for x in _ ], rotation=30, fontsize=10)
    ax.set_ylabel('TFLOPS')
    ax.legend((sc, cu), ('ISAAC', conf['lib']))
    ax.set_title('sGEMM - {}'.format(device))
    plt.savefig('bench-{}.png'.format(conf['lib']))
    plt.show()
