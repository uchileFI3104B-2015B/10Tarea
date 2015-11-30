#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import cauchy
from scipy.stats import norm
from scipy.stats import kstest
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que aproxima el espectro por un perfil de lorentz
 usando Levenberg- Marquardt'''

def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1]


def mostrar_datos(w, f, data1, data2, xlabel, ylabel, title, ylim):
    ''' función para graficar los datos'''
    ax, fig = plt.subplots()
    plt.plot(w, f, label="datos originales")
    plt.plot(data1[0], data1[1], label="modelo Perfil lorentz")
    plt.plot(data2[0], data2[1], label="modelo Gaussiana")
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_ylim(ylim)
    plt.legend(loc=4)
    plt.savefig("{}.jpg".format(title))


def linea(x, m, n):
    ''' linea recta para aproximación'''
    return x * m + n


def gauss(x, A, mu, sigma):
    '''función gaussiana para aproximación'''
    return  A * norm(loc=mu, scale=sigma).pdf(x)


def lorentz(x, A, mu, sigma):
    '''perfil de lorentz para aproximación'''
    return  A * cauchy(loc=mu, scale=sigma).pdf(x)


def modelo_l(x, A, mu, sigma, m, n):
    '''modelo utilizando perfil de lorentz  '''
    return linea(x, m, n) - lorentz(x, A, mu, sigma)


def modelo_g(x, A, mu, sigma, m, n):
    '''modelo utilizando funcion gaussiana'''
    return linea(x, m, n) - gauss(x, A, mu, sigma)


def crear_datos(param_opt, f):
    ''' crea un set de datos x,y según el modelo obtenido por curvefit'''
    x = np.linspace(6.4600e+03, 6.6596e+03, 150)
    y = f(x, param_opt[0], param_opt[1], param_opt[2], param_opt[3],
                  param_opt[4])
    return x, y


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)


def generar_prob_acumulada(x_data, y_data, a, f):
    '''funcion que recibe los datos y el parametro optimo y genera la
    distribucion acumulada del modelo'''
    xmin = np.min(x_data)
    xmax = np.max(x_data)
    y_model_sorted = np.sort(f(np.linspace(xmin, xmax, 1000), *a))
    y_data_sorted = np.sort(y_data)
    return y_model_sorted, y_data_sorted


def mostrar_prob_acum(f_d, f_m, title):
    '''grafica las funciones probabilidad acumulada'''
    ax, fig = plt.subplots()
    CDF_model = np.array([np.sum(f_m <= yy) for yy in f_d]) / len(f_m)
    N = len(f_d)
    plt.plot(f_d, np.arange(N) / N, '-^', drawstyle='steps-post', label="datos")
    plt.plot(f_d, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
    plt.plot(f_d, CDF_model, '-x', drawstyle='steps-post', label="modelo")
    fig.set_title(title)
    fig.set_xlabel( "Wavelength[Angstrom]")
    fig.set_ylabel("Probabilidad")
    plt.legend(loc=2)
    plt.savefig("{}".format(title))

# main
wlength, fnu = importar_datos("espectro.dat")
param_opt_l, pcov = curve_fit(modelo_l, wlength, fnu,
                            p0=[0.1e-16, 6550, 10, 0.04e-16/200, 1])
param_opt_g, pcov = curve_fit(modelo_g, wlength, fnu,
                            p0=[0.1e-16, 6550, 10, 0.04e-16/200, 1])
x_modelo_l, y_modelo_l = crear_datos(param_opt_l, modelo_l)
x_modelo_g, y_modelo_g = crear_datos(param_opt_g, modelo_g)
print("Parametros optimos perfil de Lorentz: Amplitud= {}, promedio = {}, sigma = {}".format(param_opt_l[0], param_opt_l[1], param_opt_l[2]))
print("Parametros optimos modelo Gaussiano: Amplitud= {}, promedio = {}, sigma = {}".format(param_opt_g[0], param_opt_g[1], param_opt_g[2]))
print("Parametros optimos recta: Pendiente = {}, coef de posicion = {}".format(param_opt_l[3],param_opt_l[4]))

mostrar_datos(wlength, fnu, [x_modelo_l, y_modelo_l], [x_modelo_g, y_modelo_g],
              "Wavelength[Angstrom]",
              "$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$",
              "modelos Perfil Lorentz y Gauss", [1.28e-16, 1.42e-16])

f_modelo_l_sorted, f_data_sorted = generar_prob_acumulada(wlength, fnu, param_opt_l, modelo_l)
f_modelo_g_sorted, f_data_sorted = generar_prob_acumulada(wlength, fnu, param_opt_g, modelo_g)
mostrar_prob_acum(f_modelo_l_sorted, f_data_sorted, "distribucion acumulada para perfil de Lorentz")
mostrar_prob_acum(f_modelo_g_sorted, f_data_sorted, "distribucion acumulada para modelo Gaussiano" )
Dn_scipy_l, conf_l = kstest(f_data_sorted, cdf, args=(f_modelo_l_sorted,))
Dn_scipy_g, conf_g = kstest(f_data_sorted, cdf, args=(f_modelo_g_sorted,))
print "Dn_scipy para perfil de lorentz   : ", Dn_scipy_l
print "Nivel de confianza para perfil de lorentz : ", conf_l

print "Dn_scipy para modelo gaussiano   : ", Dn_scipy_g
print "Nivel de confianza para modelo gaussiano : ", conf_g

plt.show()
