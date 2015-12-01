# -*- coding: utf-8 -*-
from scipy import stats
import numpy as np

"""
Created on Tue Dec 01 00:14:40 2015

@author: Administrador
"""

def gauss(A, mu, sigma ,x):
    g = A * stats.norm(loc=mu, scale=sigma).pdf(x)
    return g
    
def lorentz(A, mu, sigma, x):
    l = A * stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return l
    
def linea(x, m, n):
    return x * m + n
        
def modelo_gauss(x, A, mu, sigma, m, n):
    return linea(x, m, n) - gauss(A, mu, sigma, x)
    
def modelo_lorentz(x, A, mu, sigma, m, n):
    return linea(x, m, n) - lorentz(A, mu, sigma, x)
    
def chi_2(datos, modelo, param):
    '''
    Recibe los datos y el modelo a evaluar, y
    retorna el valor de la función chi cuadrado.
    param es un array que contiene (A, mu, sigma, m y n).
    '''
    x_m = datos[0]
    y_m = modelo(x_m, param[0], param[1], param[2], param[3], param[4])
    return np.sum((datos[1] - y_m) ** 2)

#Main
    
np.loadtxt('espectro.dat')