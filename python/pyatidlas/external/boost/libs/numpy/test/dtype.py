#!/usr/bin/env python

# Copyright Jim Bosch & Ankit Daftery 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import dtype_mod
import unittest
import numpy

class DtypeTestCase(unittest.TestCase):
    
    def assertEquivalent(self, a, b):
        return self.assert_(dtype_mod.equivalent(a, b), "%r is not equivalent to %r")

    def testIntegers(self):
        for bits in (8, 16, 32, 64):
            s = getattr(numpy, "int%d" % bits)
            u = getattr(numpy, "uint%d" % bits)
            fs = getattr(dtype_mod, "accept_int%d" % bits)
            fu = getattr(dtype_mod, "accept_uint%d" % bits)
            self.assertEquivalent(fs(s(1)), numpy.dtype(s))
            self.assertEquivalent(fu(u(1)), numpy.dtype(u))
            # these should just use the regular Boost.Python converters
            self.assertEquivalent(fs(True), numpy.dtype(s))
            self.assertEquivalent(fu(True), numpy.dtype(u))
            self.assertEquivalent(fs(int(1)), numpy.dtype(s))
            self.assertEquivalent(fu(int(1)), numpy.dtype(u))
            self.assertEquivalent(fs(long(1)), numpy.dtype(s))
            self.assertEquivalent(fu(long(1)), numpy.dtype(u))
        for name in ("bool_", "byte", "ubyte", "short", "ushort", "intc", "uintc"):
            t = getattr(numpy, name)
            ft = getattr(dtype_mod, "accept_%s" % name)
            self.assertEquivalent(ft(t(1)), numpy.dtype(t))
            # these should just use the regular Boost.Python converters
            self.assertEquivalent(ft(True), numpy.dtype(t))
            if name != "bool_":
                self.assertEquivalent(ft(int(1)), numpy.dtype(t))
                self.assertEquivalent(ft(long(1)), numpy.dtype(t))


    def testFloats(self):
        f = numpy.float32
        c = numpy.complex64
        self.assertEquivalent(dtype_mod.accept_float32(f(numpy.pi)), numpy.dtype(f))
        self.assertEquivalent(dtype_mod.accept_complex64(c(1+2j)), numpy.dtype(c))
        f = numpy.float64
        c = numpy.complex128
        self.assertEquivalent(dtype_mod.accept_float64(f(numpy.pi)), numpy.dtype(f))
        self.assertEquivalent(dtype_mod.accept_complex128(c(1+2j)), numpy.dtype(c))
        if hasattr(numpy, "longdouble"):
            f = numpy.longdouble
            c = numpy.clongdouble
            self.assertEquivalent(dtype_mod.accept_longdouble(f(numpy.pi)), numpy.dtype(f))
            self.assertEquivalent(dtype_mod.accept_clongdouble(c(1+2j)), numpy.dtype(c))
            

if __name__=="__main__":
    unittest.main()
