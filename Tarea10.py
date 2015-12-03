#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import (kstest, kstwobign)


def modelo_Gaussiano(x, m, n, A, mu, sigma):
    '''
    Retorna la diferencia entre una recta y un ajuste Gaussiano
    '''
    R = m * x + n
    L = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return R - L


def modelo_Lorentz(x, m, n, A, mu, sigma):
    '''
    Retorna la diferencia entre una recta y un ajuste Lorentziano
    '''
    R = m * x + n
    L = A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return R - L


def chi2(x , y, m, n, A, mu, sigma, f):
    y_m = f(x, m, n, A, mu, sigma)
    sigma = (y - y_m)**2
    return np.sum(sigma)


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)

# Se cargan los datos
wavelength = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])
m, n = np.polyfit(wavelength, Fnu, 1)
anzats = [m, n, 0.1e-16, 6550, 1]
x = np.linspace(min(wavelength), max(wavelength), 100)


popt1, pcov1 = curve_fit(modelo_Gaussiano, wavelength, Fnu, anzats)
m1, n1, A1, mu1, sigma1 = popt1
chi2_1 = chi2(wavelength, m1, n1, A1, mu1, sigma1, popt1, modelo_Gaussiano)
D_1, p_value1 = kstest(np.sort(Fnu), cdf,
                      args=(np.sort(modelo_Gaussiano(x, m1, n1, A1,
                                                     mu1, sigma1)),))

popt2, pcov2 = curve_fit(modelo_Lorentz, wavelength, Fnu, anzats)
m2, n2, A2, mu2, sigma2 = popt2
chi2_2 = chi2(wavelength, Fnu, m2, n2, A2, mu2, sigma2, modelo_Lorentz)
D_2, p_value2 = kstest(np.sort(Fnu), cdf,
                      args=(np.sort(modelo_Lorentz(x, m2, n2, A2,
                                                   mu2, sigma2)),))
