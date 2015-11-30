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


def recta(x, p):
    a, b = p
    return a * x + b


def gaussiana(x, p):
    A, mu, sigma = p
    y = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y


def plot(x, y):
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y, 'o')
    #ax1.plot(x, y1, label="")
    #ax1.plot(x, y2, label="")
    ax1.plot()
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    plt.legend(loc=4)
    plt.savefig("a.png")
    plt.draw()
    plt.show()


# inicializacion
x = np.linspace(-5, 5, 100)
p1 = 1, 1
p2 = 1, 0, 1
y1 = recta(x, p1)
y2 = gaussiana(x, p2)
exp = leer_archivo("espectro.dat")
x_ex = exp[:, 0]
y_ex = exp[:, 1]
# plot
plot(x_ex, y_ex)
