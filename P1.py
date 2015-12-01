# -*- coding: utf-8 -*-
from scipy import stats
import numpy as np
from scipy.optimize import (leastsq, curve_fit)

"""
Este programa extrae los datos de espectro.dat, modela el espectro
segun una perfil de Lorentz y Gauss, y ve cual es el mejor modelo
usando K-S.
"""

def gauss(A, mu, sigma, x):
    g = A * stats.norm(loc=mu, scale=sigma).pdf(x)
    return g
    
def lorentz(A, mu, sigma, x):
    l = A * stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return l
    
def linea(x, m, n):
    return x * m + n
        
def modelo_g(x, param):
    A, mu, sigma, m, n = param
    return linea(x, m, n) - gauss(A, mu, sigma, x)
    
def modelo_l(x, param):
    A, mu, sigma, m, n = param
    return linea(x, m, n) - lorentz(A, mu, sigma, x)
    
def chi_2(datos, modelo, param):
    '''
    Recibe los datos y el modelo a evaluar, y
    retorna el valor de la función chi cuadrado.
    param es un array que contiene (A, mu, sigma, m y n).
    '''
    x_m = datos[0]
    y_m = modelo(x_m, param)
    return np.sum((datos[1] - y_m) ** 2)
    
def prob_acumulada(datos, modelo):
    '''
    Esto es para hacer K-S usando scipy.
    '''
    return np.array([np.sum(modelo <= yy) for yy in datos]) / len(modelo)
    


#Main
    
d1 = np.loadtxt('espectro.dat')
l_onda = d1[:,0]
f_nu = d1[:,1]

param_opt_l, pcov = curve_fit(modelo_l, l_onda, f_nu,
                            p0)
param_opt_g, pcov = curve_fit(modelo_g, l_onda, f_nu,
                            p0)