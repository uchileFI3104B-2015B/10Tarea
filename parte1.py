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


def recta(p, x):
    a, b = p
    return a * x + b


def gaussiana(p, x):
    A, mu, sigma = p
    y = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y


def plot(x, y, y1):
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, '+', label="Datos Experimentales")
    ax1.plot(x, y1, '-', label="Ajuste Gaussiano")
    ax1.plot()
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    plt.legend(loc=4)
    plt.savefig("a.png")
    plt.draw()
    plt.show()


def modelo1(p, x):
    a, b, A, mu, sigma = p
    p1 = a, b
    p2 = A, mu, sigma
    y1 = recta(p1, x)
    y2 = gaussiana(p2, x)
    return y1 - y2


def experimental():
    ex = leer_archivo("espectro.dat")
    x_ex = ex[:, 0]
    y_ex = ex[:, 1]
    return x_ex, y_ex


def residuo(p, x_exp, y_exp):
    err = y_exp - modelo1(p, x_exp)
    return err


# inicializacion
p1 = 1e-16 , 0
p2 = 1e-16, 6550, 1
p0 = p1[0], p1[1], p2[0], p2[1], p2[2]
x_exp, y_exp = experimental()
pf = leastsq(residuo, p0, args=(x_exp, y_exp))
# plot
p_f = pf[0]
y1 = modelo1(p_f, x_exp)
plot(x_exp, y_exp, y1)
