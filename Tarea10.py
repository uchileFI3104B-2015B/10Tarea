#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' este codigo modelará la funcion del espectro
a traves de una gaussiana y del perfil de  lorenz'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.stats import kstwobign


def Recta(x, a, b):
    return a*x + b


def Gaussiana(x, A, mu, sigma):
    return A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def Lorenz(x, A, mu, sigma):
    return A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def modelo_1(x, a, b, A, mu, sigma):
    '''
    Modelación según Recta menos Gaussiana
    '''
    return Recta(x, a, b) - Gaussiana(x, A, mu, sigma)


def modelo_2(x, a, b, A, mu, sigma):
    '''
    Modelación con el perfil de Lorentz
    '''
    return Recta(x, a, b) - Lorenz(x, A, mu, sigma)


def chi_2(x, y, modelo, adivinanza):
    '''
    Error asociado
    '''
    y_modelo = modelo(x, *adivinanza)
    chi_2 = (y - y_modelo) ** 2
    return np.sum(chi_2)


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)


# Setup
long_onda = np.loadtxt("espectro.dat", usecols=[0])
Fnu = np.loadtxt("espectro.dat", usecols=[1])
x_arreglo = x = np.linspace(min(long_onda), max(long_onda), 1000)
pendiente, coef_pos = np.polyfit(long_onda, Fnu, 1)
alfa = 0.05
adivinanza = pendiente, coef_pos, 0.1e-16, 6560, 1
# Pendiente, Coeficiente posición, Amplitud, mu y sigma

# Resultados
optimo_1, covar_1 = curve_fit(modelo_1, long_onda, Fnu, adivinanza)
optimo_2, covar_2 = curve_fit(modelo_2, long_onda, Fnu, adivinanza)
chi1 = chi_2(long_onda, Fnu, modelo_1, adivinanza)
chi2 = chi_2(long_onda, Fnu, modelo_2, adivinanza)
a_1, b_1, A_1, mu_1, sigma_1 = optimo_1
a_2, b_2, A_2, mu_2, sigma_2 = optimo_2
Dn_1, prob_1 = kstest(np.sort(Fnu), cdf,
                      args=(np.sort(modelo_1(x, a_1, b_1, A_1,
                                    mu_1, sigma_1)),))
Dn_2, prob_2 = kstest(np.sort(Fnu), cdf,
                      args=(np.sort(modelo_2(x, a_2, b_2, A_2,
                                    mu_2, sigma_2)),))
ks_dist = kstwobign()
Dn_critico = ks_dist.ppf(1 - alfa) / np.sqrt(len(Fnu))

print "a1 =", a_1
print "b1 =", b_1
print "A1 =", A_1
print "mu1 =", mu_1
print "sigma1 =", sigma_1
print "chi1 =", chi1
print "dn1 =", Dn_1
print "prob1 =", prob_1
print "----°----°----°----"
print "a2 =", a_2
print "b2 =", b_2
print "A2 =", A_2
print "mu2 =", mu_2
print "sigma2 =", sigma_2
print "chi2 =", chi2
print "dn2 =", Dn_2
print "prob2 =", prob_2
print "----°----°----°----"
print "Dn critico =", Dn_critico

# Plots
plt.figure(1)
plt.plot(long_onda, Fnu, color='blue', drawstyle='steps-post',
         label="$Datos \ Flujo$")
plt.plot(x_arreglo, modelo_1(x_arreglo, * optimo_1), color='green',
         label="$Modelo \ Gaussiana$")
plt.plot(x_arreglo, modelo_2(x_arreglo, * optimo_2), color='purple',
         label="$Perfil \ de \ Lorenz$")
plt.legend(loc="lower right")
plt.xlabel("$ Wavelength [\AA]$", fontsize=15)
plt.ylabel("$ F_v [erg s^{-1} Hz^{-1} cm^{-2}]$", fontsize=15)
plt.title('$Espectro \ observado \ y \ modelos \ usados$', fontsize=15)
plt.savefig('modelacion.png')

plt.figure(2)
plt.plot(long_onda, Fnu, color='blue', drawstyle='steps-post',
         label="$Datos \ Flujo$")
plt.plot(x_arreglo, pendiente*x + coef_pos, label="$Polyfit \ orden \ 1$",
         color='green')
plt.legend(loc='lower right')
plt.xlabel("$ Wavelength [\AA]$", fontsize=15)
plt.ylabel("$ F_v [erg s^{-1} Hz^{-1} cm^{-2}]$", fontsize=15)
plt.title('$Arreglo \ lineal \ de \ los \ datos$', fontsize=15)
plt.savefig('arreglo.png')
plt.show()
