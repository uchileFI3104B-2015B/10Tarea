#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' este codigo modelará la funcion del espectro
a traves de una gaussiana y de lorentz'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.stats import kstwobign


def linea(x, a, b):
    '''funcion que modela la linea recta'''

    return a + b*x


def gaussiana(A, mu, sigma, x):
    '''funcion que modela la parte gaussiana'''

    return  A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def lorenz(A, mu, sigma, x):
    ''' funcion que modela segun lorenz'''
    return A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def modelo_1(x, a, b, A, mu, sigma):
    '''funcion que se le aplicará el fiting'''

    return linea(x, a, b) - gaussiana(A, mu, sigma, x)


def modelo_2(x, a, b, A, mu, sigma):

    ''' srgunda funcion que se le aplicara el fit'''

    return linea(x, a, b) - lorenz(A, mu, sigma, x)


datos = np.loadtxt('espectro.dat')
x_data = datos[:,0]
y_data = datos[:,1]

plt.plot(x_data,y_data,'co-')


N = 10000
a0 = 1.3e-16, 3.2996632996633799e-20, 0.1e-16, 6560, 1

a_opt, a_covar = curve_fit(modelo_1, x_data, y_data, a0)
a_opt1, a_covar1 = curve_fit(modelo_2, x_data, y_data, a0)
x = np.linspace(6400,6700,N)
plt.plot(x, modelo_1(x, *a_opt))
plt.plot(x, modelo_1(x, *a_opt1),color='deeppink')
plt.ylabel("$ Flujo [erg s-1 Hz-1 cm-2]$", fontsize=15)
plt.xlabel("$ Longitud \ de \ onda [\AA]$", fontsize=15)
plt.title('$Espectro \ observado \ y \ fit$', fontsize=15)

plt.show()
