#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################
#                                TAREA 10                                   #
#############################################################################

'''
Universidad de Chile
Facultad de Ciencias Fisicas y Matematicas
Departamento de Fisica
FI3104 Metodos Numericos para la Ciencia y la Ingenieria
Semestre Primavera 2015

Nicolas Troncoso Kurtovic
'''

from __future__ import division
import matplotlib.pyplot as p
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import os

#############################################################################
#                                                                           #
#############################################################################


def llamar_archivo(nombre):
    '''
    Llama un archivo 'nombre.dat' de dos columnas, este archivo esta
    ubicado en una carpeta 'data' que se encuentra en el mismo directorio
    que el codigo.
    '''
    cur_dir = os.getcwd()
    arch = np.loadtxt(os.path.join(cur_dir, nombre))
    x = arch[:, 0]
    y = arch[:, 1]
    return x, y


def rectagauss(x, a, b, A, mu, sigma):
    '''
    Retorna el valor de una funcion lineal menos una funcion gaussiana
    evaluado en un x dado para los parametros ingresados.
    '''
    return (a + b * x) - (A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x))


def rectalorentz(x, a, b, A, mu, sigma):
    '''
    Retorna el valor de una funcion lineal menos una funcion lorentziana
    evaluado en un x dado para los parametros ingresados.
    '''
    return (a + b * x) - (A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x))


def chi(func, x, y, parametros):
    chi2 = 0
    for i in range(len(x)):
        chi2 += (y[i] - func(x[i], *parametros))**2
    return chi2


def fitear_implementado(func, x, y, seed=False):
    '''
    Retorna los coeficientes b y a correspondientes a la ecuacion lineal que
    mejor fitea los vectores x e y segun curve_fit.
    '''
    if seed is False:
        popt, pcov = scipy.optimize.curve_fit(func, x, y)
        return popt
    else:
        popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=seed)
        return popt


def resultados_gauss(x, y, seeds=False):
    '''
    Imprime y grafica los resultados del ajuste gaussiano a los datos x, y
    '''
    # Ajustar recta-gaussiana
    popt = fitear_implementado(rectagauss, x, y, seed=seeds)
    # Escribir resultados
    print '----------------------------------------------'
    print 'f(x) = (a + b * x) - A * exp((x - mu) / sigma)'
    print 'a = ', popt[0]
    print 'b = ', popt[1]
    print 'A = ', popt[2]
    print 'mu = ', popt[3]
    print 'sigma = ', popt[4]
    print 'chi**2 = ', chi(rectagauss, x, y, popt)
    print '----------------------------------------------'
    # Graficar
    p.plot(x, rectagauss(x, *popt), 'r--')
    p.plot(x, y, 'g^')
    p.axis([6450, 6675, 1.28e-16, 1.42e-16])
    p.xlabel('Angstrom')
    p.ylabel('erg / s / Hz / cm^2')
    p.show()


def resultados_lorentz(x, y, seeds=False):
    '''
    Imprime y grafica los resultados del ajuste lorentziano a los datos x, y
    '''
    # Ajustar recta-gaussiana
    popt = fitear_implementado(rectalorentz, x, y, seed=seeds)
    # Escribir resultados
    print '----------------------------------------------'
    print 'f(x) = (a + b * x) - A / (pi * (1 + (x - mu)**2 / sigma**2))'
    print 'a = ', popt[0]
    print 'b = ', popt[1]
    print 'A = ', popt[2]
    print 'mu = ', popt[3]
    print 'sigma = ', popt[4]
    print 'chi**2 = ', chi(rectalorentz, x, y, popt)
    print '----------------------------------------------'
    # Graficar
    p.plot(x, rectalorentz(x, *popt), 'r--')
    p.plot(x, y, 'g^')
    p.axis([6450, 6675, 1.28e-16, 1.42e-16])
    p.xlabel('Angstrom')
    p.ylabel('erg / s / Hz / cm^2')
    p.show()


def comparacion(x, y, seeds=False):
    '''
    Compara los graficos de los ajustes gaussianos y lorentzianos a los
    datos x, y. Se realiza el ajuste a partir de las semillas ingresadas.
    '''
    # Ajuste
    popt1 = fitear_implementado(rectagauss, x, y, seed=seeds)
    popt2 = fitear_implementado(rectalorentz, x, y, seed=seeds)
    # Graficos
    p.plot(x, rectagauss(x, *popt1), 'r--')
    p.plot(x, rectalorentz(x, *popt2), 'g--')
    p.plot(x, y, 'y^')
    p.axis([6450, 6675, 1.28e-16, 1.42e-16])
    p.xlabel('Angstrom')
    p.ylabel('erg / s / Hz / cm^2')
    p.show()


#############################################################################
#                                                                           #
#############################################################################

# Leer datos
x, y = llamar_archivo('espectro.dat')

# Semillas para el ajuste
seeds = [9.1e-17, 7.4e-21, 1e-17, 6563., 7.]

# Resultados
resultados_gauss(x, y, seeds)
resultados_lorentz(x, y, seeds)
comparacion(x, y, seeds)
