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


def chi_cuadrado(x_data,y_data, modelo_1, a0):

    x_modelo = x_data
    y_modelo = modelo_1(x_modelo, *a0)
    chi_2 = (y_data - y_modelo) ** 2
    return np.sum(chi_2)


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)

def dn(x_data,y_data,a_opt,a_opt1):
    xmin = np.min(x_data)
    xmax = np.max(x_data)
    y_model_sorted = np.sort(modelo_1(np.linspace(xmin, xmax, 1000), *a_opt))
    y_model_sorted1 = np.sort(modelo_2(np.linspace(xmin, xmax, 1000), *a_opt1))
    y_data_sorted = np.sort(y_data)
    N = len(y_data_sorted)
    Dn_scipy, prob_scipy = kstest(y_data_sorted, cdf, args=(y_model_sorted,))
    Dn_scipy2, prob_scipy2 = kstest(y_data_sorted, cdf, args=(y_model_sorted1,))

    ks_dist = kstwobign() # two-sided, aproximacion para big N
    alpha = 0.05
    Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
    print "Dn_critico = ", Dn_critico
    print "Dn para nuestro fit = ", Dn_scipy
    print "nivel de confianza",prob_scipy
    print "Dn para nuestro fit = ", Dn_scipy2
    print "nivel de confianza =",prob_scipy2






#calculo de dn
N1 = 10000
x = np.linspace(6400,6700,N1)
datos = np.loadtxt('espectro.dat')
x_data = datos[:,0]
y_data = datos[:,1]

a0 = 1.3e-16, 3.2996632996633799e-20, 0.1e-16, 6560, 1
a_opt, a_covar = curve_fit(modelo_1, x_data, y_data, a0)
a_opt1, a_covar1 = curve_fit(modelo_2, x_data, y_data, a0)
chi1 = chi_cuadrado(x_data,y_data, modelo_1, a0)
chi2= chi_cuadrado(x_data,y_data, modelo_2, a0)


#resultados
print "chi modelo Gaussiana =",chi1
print "Chi modelo Lorenz =",chi2
dn(x_data,y_data,a_opt,a_opt1)


#plots

plt.plot(x_data,y_data,'co-')
plt.plot(x, modelo_1(x, *a_opt))
plt.plot(x, modelo_1(x, *a_opt1),color='deeppink')
plt.ylabel("$ Flujo [erg s-1 Hz-1 cm-2]$", fontsize=15)
plt.xlabel("$ Longitud \ de \ onda [\AA]$", fontsize=15)
plt.title('$Espectro \ observado \ y \ fit$', fontsize=15)
plt.show()
