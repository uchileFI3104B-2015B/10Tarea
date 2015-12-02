
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Script para encontrar modelar con perfiles de gauss y de lorentz  un segmento
de un espectro de una fuente que muestra continuo y una linea de absorcion
'''
from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats


def perfil_ajuste_gauss(x,a,b,c,mu,sigma):
    continuo = a*x+b
    g = c * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return continuo-g


def perfil_ajuste_lorentz(x,a,b,c,mu,sigma):
    continuo = a*x+b
    l = c * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return continuo-l


def ajuste_datos(func, X, Y):
    Opt, Var = curve_fit(perfil_ajuste_gauss, X, Y)
    print "o", Opt
    a, b = Opt[0], Opt[1]  #Primer ajuste
    c = 10**(-17)  # Del grafico
    aprox_mu = np.mean(X)
    aprox_sigma = np.std(X)

    Opt, Var = curve_fit(func, X, Y,p0=[a,b,c,aprox_mu,aprox_sigma])
    return Opt


def chi_cuad(func,x , y, opt):
    chi_cuadrado = np.sum((y-func(x,opt[0],opt[1],opt[2],opt[3],opt[4]))**2)
    return chi_cuadrado


def func_distribucion(modelo,datos):
    return np.array([np.sum(modelo <= yy) for yy in datos]) / (len(modelo))


# Pregunta 1

Data = np.loadtxt("espectro.dat")
Fv = Data[:,1]  #Flujo por unidad de frecuencia
L_onda = Data[:,0]  # Longitud de onda


# Obtener ajuste de parametros optimos, gaussiana

Optimos = ajuste_datos(perfil_ajuste_gauss, L_onda, Fv)

print "los parametros optimos para el perfil gaussiano son:" \
      , "\na =", Optimos[0]\
      , "\nb =", Optimos[1]\
      , "\nc =", Optimos[2]\
      , "\nmu =", Optimos[3]\
      , "\nsigma =", Optimos[4]
print "\nEl valor de X cuadrado es:"\
      , chi_cuad(perfil_ajuste_gauss,L_onda,Fv,Optimos)
# Plots
Fv_ajuste = perfil_ajuste_gauss(L_onda, Optimos[0],Optimos[1],Optimos[2],Optimos[3],Optimos[4])

plt.figure(1)
plt.clf()
plt.plot(L_onda, Fv, 'r')
plt.plot(L_onda, Fv_ajuste, "b")
plt.show()

# Obtener ajuste de parametros optimos, lorentz

Optimos2 = ajuste_datos(perfil_ajuste_lorentz, L_onda, Fv)

print "los parametros optimos para el perfil de lorentz son:" \
      , "\na =", Optimos2[0]\
      , "\nb =", Optimos2[1]\
      , "\nc =", Optimos2[2]\
      , "\nmu =", Optimos2[3]\
      , "\nsigma =", Optimos2[4]
print "\nEl valor de X cuadrado es:"\
      , chi_cuad(perfil_ajuste_lorentz,L_onda,Fv,Optimos2)
# Plots
Fv_ajuste2 = perfil_ajuste_lorentz(L_onda, Optimos2[0],Optimos2[1],Optimos2[2],Optimos2[3],Optimos2[4])

plt.figure(2)
plt.clf()
plt.plot(L_onda, Fv, 'r')
plt.plot(L_onda, Fv_ajuste2, "g")
plt.show()
