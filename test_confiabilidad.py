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
import scipy.stats
from scipy.optimize import curve_fit
import os
from scipy.stats import kstwobign
from scipy.stats import kstest

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
    '''
    Retorna el valor de una funcion lineal menos una funcion gaussiana
    evaluado en un x dado para los parametros ingresados.
    '''
    return (a + b * x) - (A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x))


def rectalorentz(x, a, b, A, mu, sigma):
    '''
    Retorna el valor de una funcion lineal menos una funcion lorentziana
    evaluado en un x dado para los parametros ingresados.
    '''
    return (a + b * x) - (A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x))


def chi(func, x, y, parametros):
    chi2 = 0
    for i in range(len(x)):
        chi2 += (y[i] - func(x[i], *parametros))**2
    return chi2


def fitear_implementado(func, x, y, seed=False):
    '''
    Retorna los coeficientes b y a correspondientes a la ecuacion lineal que
    mejor fitea los vectores x e y segun curve_fit.
    '''
    if seed is False:
        popt, pcov = scipy.optimize.curve_fit(func, x, y)
        return popt
    else:
        popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=seed)
        return popt


def funcion_cdf(y, modelo):
    '''
    Funcion de distribucion acumulada para los datos 'y' segun un modelo
    descrito por 'modelo'.
    '''
    return np.array([np.sum(modelo <= yy) for yy in y]) / len(modelo)


def c_manual(x, y, modelo, seeds, porcentaje, N):
    '''
    Determina la confiabilidad del modelo propuesto para fitear los datos
    'x' y 'y' ingresados a partir de un porcentaje de confiabilidad aceptado.
    N es el numero de datos sinteticos a crear para realizar la comparacion.
    Retorna los parametros utilizados para determinar la confiabilidad.
    '''
    # Datos a modelar
    datos_sinteticos = np.linspace(np.min(x), np.max(x), N)
    # Parametros optimos del modelo
    popt = fitear_implementado(modelo, x, y, seed=seeds)
    # Crear datos segun modelo y ordenarlos
    y_sint = np.sort(modelo(datos_sinteticos, *popt))
    # Ordenar datos experimentales
    y_datos = np.sort(y)
    # Crear CDF del modelo
    cdf = funcion_cdf(y_datos, y_sint)
    # Encontrar maxima diferencia entre CDF experimental y CDF del modelo
    n = len(y)
    max1 = np.max(cdf - np.arange(n) / n)
    max2 = np.max(np.arange(1, n + 1) / n - cdf)
    Dn = max(max1, max2)
    # Crear objeto de Kolmogorov-Smirnov
    ks_dist_g = kstwobign()
    # Porcentaje de confianza
    alpha = porcentaje / 100.
    # Distancia critica de diferencia entre CDF para el porcentaje dado
    Dn_crit = ks_dist_g.ppf(1 - alpha) / np.sqrt(n)
    # Calcular nivel de confianza para el Dn obtenido
    confianza = 1 - ks_dist_g.cdf(Dn * np.sqrt(n))
    return y_sint, y_datos, cdf, Dn, Dn_crit, confianza


def c_impl(x, y, modelo, seeds, porcentaje, N):
    '''
    Realiza lo mismo que la funcion c_manual, pero determina la distancia
    maxima entre CDFs y el intervalo de confianza con algoritmos
    predeterminados.
    '''
    # Datos a modelar
    datos_sinteticos = np.linspace(np.min(x), np.max(x), N)
    # Parametros optimos del modelo
    popt = fitear_implementado(modelo, x, y, seed=seeds)
    # Crear datos segun modelo y ordenarlos
    y_sint = np.sort(modelo(datos_sinteticos, *popt))
    # Ordenar datos experimentales
    y_datos = np.sort(y)
    # Crear CDF del modelo
    cdf = funcion_cdf(y_datos, y_sint)
    # Encontrar maxima diferencia entre CDF experimental y CDF del modelo
    n = len(y)
    # Crear objeto de Kolmogorov-Smirnov
    ks_dist_g = kstwobign()
    # Porcentaje de confianza
    alpha = porcentaje / 100.
    # Distancia critica de diferencia entre CDF para el porcentaje dado
    Dn_crit = ks_dist_g.ppf(1 - alpha) / np.sqrt(n)
    # Distancia maxima entre CDFs y nivel de confianza
    Dn, confianza = kstest(y_datos, funcion_cdf, args=(y_sint,))
    return y_sint, y_datos, cdf, Dn, Dn_crit, confianza


def graficar_cdf(y_datos, cdf):
    '''
    Grafica las cdf del modelo y de los datos.
    '''
    n = len(y_datos)
    # Graficar CDF datos experimentales
    p.plot(y_datos, np.arange(n) / n, 'y-^', drawstyle='steps-post')
    p.plot(y_datos, np.arange(1, n + 1) / n, 'y-^', drawstyle='steps-post')
    # Graficar CDF datos sinteticos del modelo
    p.plot(y_datos, cdf, 'g-x', drawstyle='steps-post')
    # Graficar
    p.xlabel('Flujo [erg cm**2 s A]')
    p.ylabel('Probabilidad')
    p.show()


#############################################################################
#                                                                           #
#############################################################################

# Leer datos
x, y = llamar_archivo('espectro.dat')
# Semillas para el ajuste
s = [9.1e-17, 7.4e-21, 1e-17, 6563., 7.]
# Numero de datos a modelar:
N = 1e+5
n = len(y)

y_sg, y_dg, cdfg, Dng_m, Dng_c, Cg = c_manual(x, y, rectagauss, s, 95, N)
y_sl, y_dl, cdfl, Dnl_m, Dnl_c, Cl = c_manual(x, y, rectalorentz, s, 95, N)

print '----------------------------------------------'
print 'Ajuste Gaussiano'
print 'Distacia Maxima entre CDF (manual) =', Dng_m
print 'Distacia critica al 5% (manual) =', Dng_c
print 'Nivel de confianza (manual) =', Cg
print 'alpha (manual) =', 1 - Cg

graficar_cdf(y_dg, cdfg)

print '----------------------------------------------'
print 'Ajuste Lorentziano'
print 'Distacia Maxima entre CDF (manual) =', Dnl_m
print 'Distacia critica al 5% (manual) =', Dnl_c
print 'Nivel de confianza (manual) =', Cl
print 'alpha (manual) =', 1 - Cl

graficar_cdf(y_dl, cdfl)

y_sg, y_dg, cdfg, Dng_m, Dng_c, Cg = c_impl(x, y, rectagauss, s, 95, N)
y_sl, y_dl, cdfl, Dnl_m, Dnl_c, Cl = c_impl(x, y, rectalorentz, s, 95, N)

print '----------------------------------------------'
print 'Ajuste Gaussiano'
print 'Distacia Maxima entre CDF (implementado) =', Dng_m
print 'Distacia critica al 5% (implementado) =', Dng_c
print 'Nivel de confianza (implementado) =', Cg
print 'alpha (implementado) =', 1 - Cg

graficar_cdf(y_dg, cdfg)

print '----------------------------------------------'
print 'Ajuste Lorentziano'
print 'Distacia Maxima entre CDF (implementado) =', Dnl_m
print 'Distacia critica al 5% (implementado) =', Dnl_c
print 'Nivel de confianza (implementado) =', Cl
print 'alpha (implementado) =', 1 - Cl

graficar_cdf(y_dl, cdfl)
