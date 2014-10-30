set logscale x

set terminal pdf
set output 'saxpy.pdf'
plot "out.dat" i 0 using 1:2 with lines title 'Naive', \
     "out.dat" i 0 using 1:3 with lines title 'Model', \
     "out.dat" i 0 using 1:4 with lines title 'Optimal'
