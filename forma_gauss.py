#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que aproxima el espectro por la suma de una recta y una funcion
Gaussiana usando Levenberg- Marquardt'''

def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1]


def mostrar_datos(data, xlabel, ylabel, title, ylim):
    ''' función para graficar los datos'''
    ax, fig = plt.subplots()
    plt.plot(data[0], data[1])
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_ylim(ylim)
    plt.savefig("{}.jpg".format(title))


def linea(x, m, n):
    ''' linea recta para aproximación'''
    return x * m + n


def gauss(x, A, mu, sigma):
    '''función gaussiana para aproximación'''
    return  A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def funcion_1(x, A, mu, sigma, m, n):
    ''' primera funcion para aproximar'''
    return linea(x, m, n) + gauss(x, A, mu, sigma)

    
# main
wlength, fnu = importar_datos("espectro.dat")
mostrar_datos([wlength, fnu], "Wavelength",
              "$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$",
              "espectro", [1.28e-16, 1.42e-16])

plt.show()
