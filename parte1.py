#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
este codigo busca modelar un espectro experimental
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import leastsq
from scipy import optimize as opt

# funciones estructurales


def leer_archivo(nombre):
    '''
    lee el archivo
    nombre debe ser un str
    '''
    datos = np.loadtxt(nombre)
    return datos


def modelo_gauss(p, x):
    a, b, A, mu, sigma = p
    y1 = a * x + b
    y2 = y = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y1 - y2


def modelo_lorentz(p, x):
    a, b, A, mu, sigma = p
    y1 = a * x + b
    y2 = A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return y1 - y2


def plot(x, y, y1, y2):
    '''
    grafica los modelos comparandolos (por separado) con los datos exp
    '''
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, '+', label="Espectro Experimental")
    ax1.plot(x, y1, '-', label="Ajuste Gaussiano")
    ax1.plot()
    ax1.set_xlabel("Longitud de Onda [$\AA$]")
    ax1.set_ylabel("Frecuencia [$erg s^{-1} Hz^{-1} cm^{-2}$]")
    plt.legend(loc=4)
    plt.savefig("gauss.png")
    fig.clf()
    ax2 = fig.add_subplot(111)
    ax2.plot(x, y, '+', label="Espectro Experimental")
    ax2.plot(x, y2, '-', label="Ajuste Lorentz")
    ax2.plot()
    ax2.set_xlabel("Longitud de Onda [$\AA$]")
    ax2.set_ylabel("Frecuencia [$erg s^{-1} Hz^{-1} cm^{-2}$]")
    plt.legend(loc=4)
    plt.savefig("lorentz.png")
    plt.draw()
    plt.show()


def experimental():
    '''
    no es muy necesaria, pero se deja para tener mas orden
    '''
    ex = leer_archivo("espectro.dat")
    x_ex = ex[:, 0]
    y_ex = ex[:, 1]
    return x_ex, y_ex


def residuo_gauss(p, x_exp, y_exp):
    err = y_exp - modelo_gauss(p, x_exp)
    return err


def residuo_lorentz(p, x_exp, y_exp):
    err = y_exp - modelo_lorentz(p, x_exp)
    return err


def chi_cuadrado(p, x, y, f):
    S = np.sum((y - f(p, x)) ** 2)
    return S

# inicializacion
p0 = 1e-16, 0, 1e-16, 6550, 1
x_exp, y_exp = experimental()
aprox_1 = leastsq(residuo_gauss, p0, args=(x_exp, y_exp))
aprox_2 = leastsq(residuo_lorentz, p0, args=(x_exp, y_exp))
# plot
p_gauss = aprox_1[0]
p_lorentz = aprox_2[0]
y1 = modelo_gauss(p_gauss, x_exp)
y2 = modelo_lorentz(p_lorentz, x_exp)
plot(x_exp, y_exp, y1, y2)
print p_gauss
print p_lorentz
print chi_cuadrado(p_gauss, x_exp, y_exp, modelo_gauss)
print chi_cuadrado(p_lorentz, x_exp, y_exp, modelo_gauss)
