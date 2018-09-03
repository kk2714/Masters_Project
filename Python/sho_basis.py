# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 00:04:54 2018

@author: Kamil
"""

from __future__ import division
from numpy.polynomial.hermite import *
import numpy
import pylab
import math
# use natural units where c = h_bar = 1
m = 1
w = 1
h_bar = 0.123


# some realistic values in SI units might be:
# mass of the electron m = 9.11e-31 kg
# Planck's constant h_bar = 1.05e-34 J s
# natural frequency of the oscillator w = 4.57e14 Hz
# more info on QHO here: https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator

pi = math.pi
x_min = -1.5
x_max = -x_min
xs = numpy.linspace(x_min,x_max,10000)

# n = 0
n = 0
psi = []
# coefficients for Hermite series, all 0s except the n-th term
herm_coeff = []
for i in range(n):
    herm_coeff.append(0)
herm_coeff.append(1)

for x in xs:
    psi.append(math.exp(-m*w*x**2/(2*h_bar)) * hermval((m*w/h_bar)**0.5 * x, herm_coeff))
# normalization factor for the wavefunction:
psi = numpy.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*h_bar))**0.25)

pylab.plot(xs, psi)

# n = 1
n = 1
psi = []
# coefficients for Hermite series, all 0s except the n-th term
herm_coeff = []
for i in range(n):
    herm_coeff.append(0)
herm_coeff.append(1)

for x in xs:
    psi.append(math.exp(-m*w*x**2/(2*h_bar)) * hermval((m*w/h_bar)**0.5 * x, herm_coeff))
# normalization factor for the wavefunction:
psi = numpy.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*h_bar))**0.25)

pylab.plot(xs, psi)

# n = 2
n = 2
psi = []
# coefficients for Hermite series, all 0s except the n-th term
herm_coeff = []
for i in range(n):
    herm_coeff.append(0)
herm_coeff.append(1)

for x in xs:
    psi.append(math.exp(-m*w*x**2/(2*h_bar)) * hermval((m*w/h_bar)**0.5 * x, herm_coeff))
# normalization factor for the wavefunction:
psi = numpy.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*h_bar))**0.25)

pylab.plot(xs, psi)

pylab.xlim(xmax=x_max, xmin=x_min)
pylab.xlabel("$q$")
#pylab.ylabel("$\psi_{" + str(n) + "}(x)$", size=18)
pylab.ylabel("$\psi(q)$")
pylab.legend(('n=0', 'n=1', 'n=2'),
           shadow=True, loc=(0.80, 0.20))
pylab.title("Quantum Harmonic Oscillator Wavefunctions")
pylab.show()