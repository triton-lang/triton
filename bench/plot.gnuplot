
set terminal pdf
set output 'bench.pdf'

set xlabel 'N'
set ylabel 'Bandwidth (GB/s)'
set key top left
stats "out.dat" nooutput

set logscale x
do for [i=1:STATS_blocks]{
plot "out.dat" index (i-1) using 1:2 with lines title 'ViennaCL', \
     "out.dat" index (i-1) using 1:3 with lines title 'Model', \
     "out.dat" index (i-1) using 1:4 with lines title 'Optimal', \
     "out.dat" index (i-1) using 1:5 with lines title 'CuBLAS'
}
