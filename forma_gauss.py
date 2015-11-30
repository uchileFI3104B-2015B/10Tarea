#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import kstest
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que aproxima el espectro por la suma de una recta y una funcion
Gaussiana usando Levenberg- Marquardt'''

def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1]


def mostrar_datos(w, f, data, xlabel, ylabel, title, ylim):
    ''' función para graficar los datos'''
    ax, fig = plt.subplots()
    plt.plot(w, f, label="datos origininales")
    plt.plot(data[0], data[1], label="modelo")
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_ylim(ylim)
    plt.legend(loc=3)
    plt.savefig("{}.jpg".format(title))


def linea(x, m, n):
    ''' linea recta para aproximación'''
    return x * m + n


def gauss(x, A, mu, sigma):
    '''función gaussiana para aproximación'''
    return  A * norm(loc=mu, scale=sigma).pdf(x)


def funcion_1(x, A, mu, sigma, m, n):
    ''' primera funcion para aproximar'''
    return linea(x, m, n) - gauss(x, A, mu, sigma)


def crear_datos(param_opt):
    ''' crea un set de datos x,y según el modelo obtenido por curvefit'''
    x = np.linspace(6.4600e+03, 6.6596e+03, 150)
    y = funcion_1(x, param_opt[0], param_opt[1], param_opt[2], param_opt[3],
                  param_opt[4])
    return x, y


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)


def generar_prob_acumulada(x_data, y_data, a):
    '''funcion que recibe los datos y el parametro optimo y genera la
    distribucion acumulada del modelo'''
    xmin = np.min(x_data)
    xmax = np.max(x_data)
    y_model_sorted = np.sort(funcion_1(np.linspace(xmin, xmax, 1000), *a))
    y_data_sorted = np.sort(y_data)
    return y_model_sorted, y_data_sorted


def mostrar_prob_acum(f_d, f_m):
    '''grafica las funciones probabilidad acumulada'''
    ax, fig = plt.subplots()
    CDF_model = np.array([np.sum(f_m <= yy) for yy in f_d]) / len(f_m)
    N = len(f_d)
    plt.plot(f_d, np.arange(N) / N, '-^', drawstyle='steps-post')
    plt.plot(f_d, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
    plt.plot(f_d, CDF_model, '-x', drawstyle='steps-post')


# main
wlength, fnu = importar_datos("espectro.dat")
param_opt, pcov = curve_fit(funcion_1, wlength, fnu,
                            p0=[0.1e-16, 6550, 10, 0.04e-16/200, 1])
x_modelo, y_modelo = crear_datos(param_opt)
print("Parametros optimos gauss: Amplitud= {}, promedio{}, sigma{}".format(param_opt[0], param_opt[1], param_opt[2]))
print("Parametros optimos recta: Pendiente={}, coef de posicion={}".format(param_opt[3],param_opt[4]))
mostrar_datos(wlength, fnu, [x_modelo, y_modelo],  "Wavelength",
              "$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$",
              "Modelo recta + Gaussiana", [1.28e-16, 1.42e-16])
f_modelo_sorted, f_data_sorted = generar_prob_acumulada(wlength, fnu, param_opt)
mostrar_prob_acum(f_modelo_sorted, f_data_sorted)
Dn_scipy, conf = kstest(f_data_sorted, cdf, args=(f_modelo_sorted,))
print "Dn_scipy   : ", Dn_scipy
print "Nivel de confianza : ", conf
plt.show()
