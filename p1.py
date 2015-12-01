# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import (leastsq, curve_fit)
from scipy import optimize as opt
from scipy.stats import kstest
import os

'''
Este script modela 5 parametros a la vez a partir de los datos de
espectro.dat, al pltear se observa un segmento del espectro de una fuente
que muesta continuo (con una leve pendiente) y una linea de absorcion, se modela
el primero con una linea recta (2parametros) y el segundo con una gaussiana
o con un perfil de lorentz
'''


def datos():
    '''
    Descarga los datos de un archivo y los retorna en columnas
    '''
    lambda1 = np.loadtxt('espectro.dat', usecols=(-2,))
    flujo = np.loadtxt('espectro.dat', usecols=(-1,))
    return lambda1, flujo


def linea(x, a, b):
    '''
    modela continuo a traves de una linea recta
    '''
    return a + b * x


def gaussiana(x, A, mu, sigma):
    '''
    modela linea de absorsion a traves de una gaussiana
    '''
    return A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)

def Lorentz(x, A, mu, sigma):
    '''
    modela linea de absorsion a traves del perfil de loretnz
    '''
    return A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def espectroGauss(x, a, b , A, mu, sigma):
    '''
    modela espectro total, con continuo y linea de absorcion con gauss
    '''
    return linea(x, a, b) - gaussiana(x, A , mu, sigma)

def espectroLorentz(x, a, b , A, mu, sigma):
    '''
    modela espectro total, con continuo y linea de absorcion con lorentz
    '''
    return linea(x, a, b) - Lorentz(x, A , mu, sigma)

def mejor_fit(x_data, y_data, a0):
    '''
    crea ajuste para la funcion espectro
    '''
    agauss_opt, a_covar = curve_fit(espectroGauss, x_data, y_data, a0)
    alorentz_opt, a_covar = curve_fit(espectroLorentz, x_data, y_data, a0)
    return agauss_opt, alorentz_opt


def grafico(x_data, y_data, ag_opt, al_opt):
    '''
    grafica los datos del archivo espectro.dat y el ajuste hecho
    '''

    plt.figure(1)
    plt.plot(x_data, y_data, 'c-o', label='Datos Observados')
    plt.plot(x_data, espectroGauss(x_data, *ag_opt), label='Ajuste')
    plt.xlabel('Longitud de Onda $[\AA]$')
    plt.ylabel('$F_{v} [erg s-1 Hz-1 cm-2]$')
    plt.title('Espectro de datos con ajuste Lineal y Gauss')
    plt.legend()

    plt.figure(2)
    plt.plot(x_data, y_data, 'c-o', label='Datos Observados')
    plt.plot(x_data, espectroLorentz(x_data, *al_opt), 'm', label='Ajuste')
    plt.xlabel('Longitud de Onda $[\AA]$')
    plt.ylabel('$F_{v} [erg s-1 Hz-1 cm-2]$')
    plt.title('Espectro de datos con ajuste Lineal y Gauss')
    plt.legend()
    plt.show()



# Main

longitud, flujo = datos()
A0 = 1e-20, 1 * 1e-16, 1 * 1e-16, 6570, 1
AG_opt, AL_opt = mejor_fit(longitud, flujo, A0)
grafico(longitud, flujo, AG_opt, AL_opt)
