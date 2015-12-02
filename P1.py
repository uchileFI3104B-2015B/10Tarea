# -*- coding: utf-8 -*-
from __future__ import division
from scipy import stats
import numpy as np
from scipy.optimize import (leastsq, curve_fit)
import matplotlib.pyplot as plt


"""
Este programa extrae los datos de espectro.dat, modela el espectro
segun una perfil de Lorentz y Gauss, y ve cual es el mejor modelo
usando K-S.
"""

def gauss(A, mu, sigma, x):
    return A * stats.norm(loc=mu, scale=sigma).pdf(x)
    
def lorentz(A, mu, sigma, x):
    return A * stats.cauchy(loc=mu, scale=sigma).pdf(x)
    
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
    
def cdf(datos, modelo):
    '''
    Esto es para hacer K-S usando scipy.
    '''
    return np.array([np.sum(modelo <= yy) for yy in datos]) / len(modelo)

def def_datos(param_opt, modelo):
    '''
    Crea un set de datos x,y según el modelo obtenido por curvefit.
    '''
    x = np.linspace(6.4600e+03, 6.6596e+03, 122)
    y = modelo(x, param_opt[0], param_opt[1], param_opt[2], param_opt[3],
                  param_opt[4])
    return x, y   

def graficar_datos(l, f, datosl, datosg, xlabel, ylabel, title, ylim):
    ax, fig = plt.subplots()
    plt.plot(l, f, label="Datos originales")
    plt.plot(datosl[0], datosl[1], label="Modelo Perfil Lorentz")
    plt.plot(datosg[0], datosg[1], label="Modelo Gaussiano")
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_ylim(ylim)
    plt.legend(loc=4)
    plt.savefig("GraficoP1.jpg")
    
def generar_dist_acumulada(datos_x, datos_y, a, modelo):
    '''
    Genera la distribucion para el uso de K-S
    '''
    x_min = np.min(datos_x)
    x_max = np.max(datos_x)
    y_modelo_distribuido = np.sort(modelo(np.linspace(x_min, x_max, 122), *a))
    datos_y_distribuidos = np.sort(datos_y)
    return y_modelo_distribuido, datos_y_distribuidos
    
    
def graficar_prob(datos, modelo, titulo, n_archivo):
    ax, fig = plt.subplots()    
    CDF_model = np.array([np.sum(modelo <= yy) for yy in datos]) / len(modelo)
    N = len(datos)
    plt.plot(datos, np.arange(N) / N, '-^', drawstyle='steps-post', label="datos")
    plt.plot(datos, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
    plt.plot(datos, CDF_model, '-x', drawstyle='steps-post', label="modelo")
    fig.set_title(titulo)
    fig.set_xlabel( "Wavelength[Angstrom]")
    fig.set_ylabel("Probabilidad")
    plt.legend(loc=2)
    plt.savefig("{}.jpg".format(n_archivo))

#Main
    
d1 = np.loadtxt('espectro.dat')
l_onda = d1[:,0]
f_nu = d1[:,1]

param_opt_l, pcov = curve_fit(modelo_lorentz, l_onda, f_nu,
                            p0=[0.1e-16,6560,5,0.04e-16/200,1])
param_opt_g, pcov = curve_fit(modelo_gauss, l_onda, f_nu,
                            p0=[0.1e-16,6560,5,0.04e-16/200,1])
                            
chi_l = chi_2([l_onda, f_nu], modelo_lorentz, param_opt_l) #Se sacan los chi_2
chi_g = chi_2([l_onda, f_nu], modelo_gauss, param_opt_g)

print('''Parametros optimos perfil de Lorentz: Amplitud= {},
      mu = {}, sigma = {}'''.format(param_opt_l[0],
                                          param_opt_l[1], param_opt_l[2]))
print('''Parametros optimos recta (Lorentz): Pendiente = {},
      coef de posicion = {}'''.format(param_opt_l[3],param_opt_l[4]))
print("Chi cuadrado (Lorentz)= ", chi_l)
print(" ")
print('''Parametros optimos modelo Gaussiano: Amplitud= {},
      mu = {}, sigma = {}'''.format(param_opt_g[0],
                                          param_opt_g[1], param_opt_g[2]))
print('''Parametros optimos recta (Gauss): Pendiente = {},
      coef de posicion = {}'''.format(param_opt_g[3],param_opt_g[4]))
print("Chi cuadrado (Gauss) = ", chi_g)
print(" ")      

x_modelo_l, y_modelo_l = def_datos(param_opt_l, modelo_lorentz)
x_modelo_g, y_modelo_g = def_datos(param_opt_g, modelo_gauss)

graficar_datos(l_onda, f_nu, [x_modelo_l, y_modelo_l], [x_modelo_g, y_modelo_g],
               "Longitud Onda [Angstrom]",
               "$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$",
               "Modelos Perfil Lorentz y Gauss", [1.28e-16, 1.42e-16])

# K-S

f_modelo_l_sorted, f_data_sorted = generar_dist_acumulada(l_onda, f_nu,
                                                          param_opt_l,
                                                          modelo_lorentz)
f_modelo_g_sorted, f_data_sorted = generar_dist_acumulada(l_onda, f_nu,
                                                          param_opt_g,
                                                           modelo_gauss)
                                                          
Dn_scipy_l, conf_l = stats.kstest(f_data_sorted, cdf, args=(f_modelo_l_sorted,))
Dn_scipy_g, conf_g = stats.kstest(f_data_sorted, cdf, args=(f_modelo_g_sorted,))
            
print "Dn_scipy para perfil de lorentz   : ", Dn_scipy_l
print "Nivel de confianza para perfil de lorentz : ", conf_l

print "Dn_scipy para modelo gaussiano   : ", Dn_scipy_g
print "Nivel de confianza para modelo gaussiano : ", conf_g

graficar_prob(f_data_sorted, f_modelo_l_sorted, 
            "Probabilidad acumulada para perfil de Lorentz",
            "P_Lorentz")

graficar_prob(f_data_sorted, f_modelo_g_sorted, 
            "Probabilidad acumulada para Gaussiana",
            "P_Gauss")
            
plt.show()

