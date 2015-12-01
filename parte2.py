#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
este codigo busca verificar si los modelos de gauss y/o lorentz son adecuados
para los datos experimentales.
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import leastsq
from scipy import optimize as opt
from scipy.stats import kstest

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


def experimental():
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


def cdf(datos, modelo):
    return np.array([np.sum(modelo <= yy) for yy in datos]) / len(modelo)


def prueba_ks(x_exp, y_exp, modelo, p_opt):
    xmin = np.min(x_exp)
    xmax = np.max(x_exp)
    y_modelo_ord = np.sort(modelo(p_opt, np.linspace(xmin, xmax, 1000)))
    y_exp_ord = np.sort(y_exp)
    Dn, prob = kstest(y_exp_ord, cdf, args=(y_modelo_ord,))
    return Dn, prob


def graficos_ks(x_exp, y_exp, modelo, p_opt, titulo):
    xmin = np.min(x_exp)
    xmax = np.max(x_exp)
    y_modelo_ord = np.sort(modelo(p_opt, np.linspace(xmin, xmax, 1000)))
    y_exp_ord = np.sort(y_exp)
    CDF = cdf(y_exp_ord, y_modelo_ord)
    N = len(y_exp_ord)
    ax, fig = plt.subplots()
    plt.plot(y_exp_ord, np.arange(N) / N, '-^', drawstyle='steps-post',
             label="Datos")
    plt.plot(y_exp_ord, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
    plt.plot(y_exp_ord, CDF, '-+', drawstyle='steps-post', label="Modelo")
    fig.set_xlabel("Longitud de Onda [$\AA$]")
    fig.set_ylabel("Probabilidad")
    plt.legend(loc=2)
    plt.savefig("{}".format(titulo))
    plt.show()


# main
p0 = 1e-16, 0, 1e-16, 6550, 1
x_exp, y_exp = experimental()
aprox_1 = leastsq(residuo_gauss, p0, args=(x_exp, y_exp))
aprox_2 = leastsq(residuo_lorentz, p0, args=(x_exp, y_exp))
opt1 = aprox_1[0]
opt2 = aprox_2[0]
graficos_ks(x_exp, y_exp, modelo_gauss, opt1, "prob_gauss")
graficos_ks(x_exp, y_exp, modelo_lorentz, opt2, "prob_lorentz")
print prueba_ks(x_exp, y_exp, modelo_gauss, opt1)
print prueba_ks(x_exp, y_exp, modelo_lorentz, opt2)
