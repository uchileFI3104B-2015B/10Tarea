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
    return (a + b * x) - (A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x))


def fitear_implementado(func, x, y):
    '''
    Retorna los coeficientes b y a correspondientes a la ecuacion lineal que
    mejor fitea los vectores x e y segun curve_fit.
    '''
    popt, pcov = curve_fit(func, x, y)
    return popt


def muestra_sintetica(x, y):
    '''
    A partir de los datos x, y, crea una muestra del mismo largo de estos
    arreglos utilizando sus mismos valores.
    '''
    n = len(x)
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        j = np.random.randint(n)
        xs[i] = x[j]
        ys[i] = y[j]
    return xs, ys


def intervalo_confianza(func, x, y, N, porcentaje):
    '''
    Determina el intervalo de confianza con un portentaje dado a partir de N
    calculos independientes, todo a partir de los arreglos x, y.
    '''
    b = np.zeros(N)
    p = (100. - porcentaje) / 100.
    for i in range(N):
        xs, ys = muestra_sintetica(x, y)
        popt = fitear_implementado(func, x, y)
        b[i] = popt[0]
    b = np.sort(b, kind='mergesort')
    ll = b[int(N * p / 2.)]
    ul = b[int(N * (1. - p / 2.))]
    return ll, ul


#############################################################################
#                                                                           #
#############################################################################

x, y = llamar_archivo('espectro.dat')

p.plot(x, y, 'g')
p.plot(x, y, 'g^')
p.axis([6450, 6675, 1.28e-16, 1.42e-16])
p.xlabel('Angstrom')
p.ylabel('erg / s / Hz / cm^2')
p.show()
