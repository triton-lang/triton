set logscale x

set term wxt 1
plot "out.dat" i 0 using 1:2 with lines title 'Naive', \
     "out.dat" i 0 using 1:3 with lines title 'Model', \
     "out.dat" i 0 using 1:4 with lines title 'Optimal'
     
set term wxt 2
plot "out.dat" i 1 using 1:2 with lines title 'Naive', \
     "out.dat" i 1 using 1:3 with lines title 'Model', \
     "out.dat" i 1 using 1:4 with lines title 'Optimal'
