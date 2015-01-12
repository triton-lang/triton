# Copyright Jim Bosch 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy
import gaussian

mu = numpy.zeros(2, dtype=float)
sigma = numpy.identity(2, dtype=float)
sigma[0, 1] = 0.15
sigma[1, 0] = 0.15

g = gaussian.bivariate_gaussian(mu, sigma)

r = numpy.linspace(-40, 40, 1001)
x, y = numpy.meshgrid(r, r)

z = g(x, y)

s = z.sum() * (r[1] - r[0])**2
print "sum (should be ~ 1):", s

xc = (z * x).sum() / z.sum()
print "x centroid (should be ~ %f): %f" % (mu[0], xc)

yc = (z * y).sum() / z.sum()
print "y centroid (should be ~ %f): %f" % (mu[1], yc)

xx = (z * (x - xc)**2).sum() / z.sum()
print "xx moment (should be ~ %f): %f" % (sigma[0,0], xx)

yy = (z * (y - yc)**2).sum() / z.sum()
print "yy moment (should be ~ %f): %f" % (sigma[1,1], yy)

xy = 0.5 * (z * (x - xc) * (y - yc)).sum() / z.sum()
print "xy moment (should be ~ %f): %f" % (sigma[0,1], xy)
