#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' este codigo modelará la funcion del espectro
a traves de una gaussiana y de lorentz'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.stats import kstwobign

datos = np.loadtxt('espectro.dat')
#print datos
x_data = datos[:,0]
y_data = datos[:,1]
plt.plot(x_data,y_data)
plt.show()
