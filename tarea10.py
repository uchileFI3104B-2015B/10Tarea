#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import (kstest, kstwobign)

'''
En este código se busca aproximar algo
'''


def recta(parametros, x):
    m, n = parametros
    return m * x + n


def Gaussiana(parametros, x):
    A, mu, sigma = parametros
    return A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def Lorentz(parametros, x):
    A, mu, sigma = parametros
    return A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def modelo_1(x, m, n, A, mu, sigma):
    '''
    Retorna la función correspondiente al primer modelo, es decir, una recta
    menos una funcion gaussiana
    '''
    parametros_recta = [m, n]
    parametros_gauss = [A, mu, sigma]
    return recta(parametros_recta, x) - Gaussiana(parametros_gauss, x)


def modelo_2(x, m, n, A, mu, sigma):
    '''
    Retorna la función correspondiente al primer modelo, es decir, una recta
    menos un perfil de Lorentz
    '''
    parametros_recta = [m, n]
    parametros_lorentz = [A, mu, sigma]
    return recta(parametros_recta, x) - Lorentz(parametros_lorentz, x)


def chi2(data, parametros_modelo, funcion_modelo):
    x_datos = data[0]
    y_datos = data[1]
    m, n, A, mu, sigma = parametros_modelo
    y_modelo = funcion_modelo(x_datos, m, n, A, mu, sigma)
    chi2 = (y_datos - y_modelo)**2
    return np.sum(chi2)


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)

# Main
# Datos
wavelength = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])
pendiente_adivin = (Fnu[-1] - Fnu[0]) / (wavelength[-1] - wavelength[0])
adivinanza = [1e-20, 1.39*(1e-16), 0.1*(1e-16), 6570, 1]  # m, n, A, mu, sigma
x = np.linspace(min(wavelength), max(wavelength), 100)

# Modelo 1
param_optimo1, param_covar1 = curve_fit(modelo_1, wavelength, Fnu, adivinanza)
m1, n1, A1, mu1, sigma1 = param_optimo1
chi2_1 = chi2([wavelength, Fnu], param_optimo1, modelo_1)
Dn_1, prob_1 = kstest(np.sort(Fnu), cdf,
                      args=(np.sort(modelo_1(x, m1, n1, A1, mu1, sigma1)),))

print 'Primer modelo: recta menos gaussiana'
print 'Pendiente               :', m1
print 'Coeficiente de posicion :', n1
print 'Amplitud                :', A1
print 'mu                      :', mu1
print 'sigma                   :', sigma1
print 'Chi2                    :', chi2_1
print "Dn                      : ", Dn_1
print "Nivel de confianza      : ", prob_1
print ''

# Modelo 2
param_optimo2, param_covar2 = curve_fit(modelo_2, wavelength, Fnu, adivinanza)
m2, n2, A2, mu2, sigma2 = param_optimo2
chi2_2 = chi2([wavelength, Fnu], param_optimo2, modelo_2)
Dn_2, prob_2 = kstest(np.sort(Fnu), cdf,
                      args=(np.sort(modelo_2(x, m2, n2, A2, mu2, sigma2)),))
print 'Segundo modelo: recta menos perfil de Lorentz'
print 'Pendiente               :', m2
print 'Coeficiente de posicion :', n2
print 'Amplitud                :', A2
print 'mu                      :', mu2
print 'sigma                   :', sigma2
print 'Chi2                    :', chi2_2
print "Dn                      : ", Dn_2
print "Nivel de confianza      : ", prob_2
print ''

# Calculo Dn critico
ks_dist = kstwobign()
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(len(Fnu))
print "Dn_critico = ", Dn_critico


# Plots

plt.figure(1)
plt.plot(wavelength, Fnu, color='darkcyan', drawstyle='steps-post',
         label='Datos')
plt.plot(x, modelo_1(x, m1, n1, A1, mu1, sigma1), color='blue',
         label='Modelo 1 (Gaussiana)')
plt.plot(x, modelo_2(x, m2, n2, A2, mu2, sigma2), color='fuchsia',
         label='Modelo 2 (Lorentz)')
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.ylabel('$F_v$[ergs$^{-1}$Hz$^{-1}$cm$^{-2}$]', fontsize=16)
plt.legend(loc='lower right')
plt.show()
