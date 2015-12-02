#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy.stats import kstwobign as kstw  # Para parte 2
from scipy.stats import kstest  # Para parte 2

############################################
#    Definicion de funciones a utilizar    #
############################################


def recta(p, x):
    '''
    La recta para ambas ecuaciones
    '''
    a, b = p
    return a * x + b


def gaussiana(p, x):
    '''
    La gaussiana, usar stats.norm. Toma amplitud, c y varianza.
    '''
    A, mu, sigma = p
    return A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def lorentz(p, x):
    '''
    toma amplitud, c y var, ademas de recta
    '''
    A, mu, sigma = p
    return A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def modelo_gauss(p, x):
    '''
    Tira la recta menos la gaussiana
    '''
    precta = p[0], p[1]
    pgauss = p[2], p[3], p[4]
    return recta(precta, x) - gaussiana(pgauss, x)


def modelo_lorentz(p, x):
    '''
    con p parametros
    Tira recta menos perfil de lorentz
    '''
    precta = p[0], p[1]
    plorentz = p[2], p[3], p[4]
    return recta(precta, x) - lorentz(plorentz, x)


def minimiza_gauss(x, m, n, A, mu, sigma):
    '''
    Tiene que recibir 5 cosas
    '''
    p = m, n, A, mu, sigma
    return modelo_gauss(p, x)


def minimiza_lorentz(x, m, n, A, mu, sigma):
    p = m, n, A, mu, sigma
    return modelo_lorentz(p, x)


def chi2(x_data, y_data, modelo, p):

    x_modelo = x_data
    y_modelo = modelo(p, x_modelo)
    #  print 'y_modelo', y_modelo
    chi_2 = (y_data - y_modelo) ** 2
    callable01 = np.sum(chi_2)
    return callable01


def plotear(x, y, nombre):
    '''
    Le das el con el que quieres guardar
    la imagen. En el y van el (fnu, y_gauss, y_lorentz).
    '''
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.plot(x, y[0], color='b', drawstyle='steps-post', label="Datos")
    plt.plot(x, y[1], color='r', label="Ajuste Gauss")
    plt.plot(x, y[2], color='g', label="Ajuste Lorentz")
    plt.xlabel('Longitud de Onda [$\AA$]', fontsize=18)
    plt.ylabel(' Flujo / s $\\frac{erg}{s Hz cm**2}$', fontsize=18)
    plt.title('Wavelength vs Flux', fontsize=22)  # diagrama de hubble xd
    plt.legend(loc=4)
    plt.savefig(nombre+'.eps')
    plt.grid('on')
    plt.draw()
    plt.show()


def cdf(data, modelo):  # standard
    return np.array([np.sum(modelo <= yy) for yy in data]) / len(modelo)


#############################################
#         Empieza el programa, Parte I      #
#############################################

datos = np.loadtxt("espectro.dat")
wavelength = datos[:, 0]
fnu = datos[:, 1]

m, n = np.polyfit(wavelength, fnu, 1)  # adivinanza recta
adivinanza = m, n, 0.1e-16, 6560, 10  # adivinanza: recta y A, mu, sigma
optimo = []
############################
#    En Gauss y Lorentz    #
############################

resultado_gauss = curve_fit(minimiza_gauss, wavelength, fnu, adivinanza)
resultado_lorentz = curve_fit(minimiza_lorentz, wavelength, fnu, adivinanza)
optimo = resultado_gauss[0], resultado_lorentz[0]
print optimo
covar = resultado_gauss[1], resultado_lorentz[1]  # Para parte 2
# Chi2
chi2_gauss = chi2(wavelength, fnu, modelo_gauss, optimo[0])
chi2_lorentz = chi2(wavelength, fnu, modelo_lorentz, optimo[1])

print "Parametros Optimos: Gauss, Lorentz =  ", optimo
print "$Chi**2$ Gauss, Lorentz =  ", chi2_gauss, chi2_lorentz

############################
#     Smirnov Kolmogorov   #
############################
x = np.linspace(min(wavelength), max(wavelength), 100)

dn1, conf1 = kstest(np.sort(fnu), cdf,
                    args=(np.sort(modelo_gauss(optimo[0], x)),))

dn2, conf2 = kstest(np.sort(fnu), cdf,
                    args=(np.sort(modelo_lorentz(optimo[1], x)),))

alpha = 0.05
dnc = kstw.ppf(1 - alpha) / len(fnu)**.5

print 'dn (gauss, lorentz, critico) =  ', dn1, dn2, dnc
print 'confianza (gauss, lorentz) =  ', conf1, conf2

#####################
#       PLOTS       #
#####################

y_gauss = modelo_gauss(optimo[0], wavelength)
y_lorentz = modelo_lorentz(optimo[1], wavelength)
y = fnu, y_gauss, y_lorentz

plotear(wavelength, y, 'comparar')
