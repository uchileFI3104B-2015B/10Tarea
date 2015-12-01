#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import (leastsq, curve_fit)

'''
En este código se busca aproximar algo
'''

wavelength = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])

plt.figure(1)
plt.plot(wavelength, Fnu, color='darkcyan', drawstyle='steps-post',
         linewidth=2.0)
plt.xlabel('Wavelength [$\AA$]', fontsize=14)
plt.ylabel('$F_v  [ergs^{-1}Hz^{-1}cm^{-2}]$', fontsize=14)
plt.show()
