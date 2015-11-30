#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import cauchy
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que aproxima el espectro por un perfil de lorentz
 usando Levenberg- Marquardt'''

def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1]


def mostrar_datos(w, f, data, xlabel, ylabel, title, ylim):
    ''' función para graficar los datos'''
    ax, fig = plt.subplots()
    plt.plot(w, f, label="datos originales")
    plt.plot(data[0], data[1], label="modelo")
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_ylim(ylim)
    plt.savefig("{}.jpg".format(title))


def linea(x, m, n):
    ''' linea recta para aproximación'''
    return x * m + n


def lorentz(x, A, mu, sigma):
    '''función gaussiana para aproximación'''
    return  A * cauchy(loc=mu, scale=sigma).pdf(x)


def funcion_1(x, A, mu, sigma, m, n):
    ''' primera funcion para aproximar'''
    return linea(x, m, n) - lorentz(x, A, mu, sigma)


def crear_datos(param_opt):
    ''' crea un set de datos x,y según el modelo obtenido por curvefit'''
    x = np.linspace(6.4600e+03, 6.6596e+03, 150)
    y = funcion_1(x, param_opt[0], param_opt[1], param_opt[2], param_opt[3],
                  param_opt[4])
    return x, y


# main
wlength, fnu = importar_datos("espectro.dat")
param_opt, pcov = curve_fit(funcion_1, wlength, fnu,
                            p0=[0.1e-16, 6550, 10, 0.04e-16/200, 1])
x_modelo, y_modelo = crear_datos(param_opt)
print("Parametros optimos perfil de Lorentz: Amplitud= {}, promedio{}, sigma{}".format(param_opt[0], param_opt[1], param_opt[2]))
print("Parametros optimos recta: Pendiente={}, coef de posicion={}".format(param_opt[3],param_opt[4]))
mostrar_datos(wlength, fnu, [x_modelo, y_modelo],  "Wavelength",
              "$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$",
              "modelo Perfil Lorentz", [1.28e-16, 1.42e-16])

plt.show()
