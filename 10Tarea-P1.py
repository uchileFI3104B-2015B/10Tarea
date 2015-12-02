
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
    '''
    Modela el continuo con la linea de absorcion como funcion lineal
    menos una funcion gaussiana, siendo a y b los parametros de la funcion
    lineal ax+b, y c, mu, sigma los parametros de la funcion gaussiana
    con c siendo la amplitud.
    '''
    continuo = a*x+b
    g = c * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return continuo-g


def perfil_ajuste_lorentz(x,a,b,c,mu,sigma):
    '''
    Modela el continuo con la linea de absorcion como funcion lineal
    menos una funcion lorentziana, siendo a y b los parametros de la funcion
    lineal ax+b, y c, mu, sigma los parametros de la funcion gaussiana
    con c siendo la amplitud.
    '''
    continuo = a*x+b
    l = c * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return continuo-l


def ajuste_datos(func, X, Y):
    '''
    Entrega los parametros optimos dados los datos X, Y y la funcion
    para hacer el ajuste
    '''
    Opt, Var = curve_fit(perfil_ajuste_gauss, X, Y)  #Primer ajuste

    a, b = Opt[0], Opt[1]
    c = 10**(-17)  # se obtiene del grafico
    aprox_mu = np.mean(X)
    aprox_sigma = np.std(X)

    Opt, Var = curve_fit(func, X, Y,p0=[a,b,c,aprox_mu,aprox_sigma])
    return Opt


def chi_cuad(func,x , y, opt):
    '''
    Retorna Chi cuadrado del ajuste
    '''
    chi_cuadrado = np.sum((y-func(x,opt[0],opt[1],opt[2],opt[3],opt[4]))**2)
    return chi_cuadrado


def func_distribucion(modelo,datos):
    '''
    Funcion de distribucion para el ks test
    '''
    return np.array([np.sum(modelo <= yy) for yy in datos]) / (len(modelo))


# Pregunta 1

Data = np.loadtxt("espectro.dat")
Fv = Data[:,1]  #Flujo por unidad de frecuencia
L_onda = Data[:,0] # Longitud de onda


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
plt.step(L_onda, Fv, 'r')
plt.plot(L_onda, Fv_ajuste, "b")
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel(r'$F_\nu \, [ergs \, s^{-1}\, Hz^{-1}\, cm^{-2}]$')
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
plt.step(L_onda, Fv, 'r')
plt.plot(L_onda, Fv_ajuste2, "g")
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel(r'$F_\nu \, [ergs \, s^{-1}\, Hz^{-1}\, cm^{-2}]$')
plt.show()

plt.figure(3)
plt.clf()
plt.step(L_onda, Fv, 'r')
plt.plot(L_onda, Fv_ajuste, "b", label="Gaussiano")
plt.plot(L_onda, Fv_ajuste2, "g", label="Lorentziano")
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel(r'$F_\nu \, [ergs \, s^{-1}\, Hz^{-1}\, cm^{-2}]$')
plt.show()


# Pregunta 2

# Modelo gaussiano
L_onda_max = np.max(L_onda)
L_onda_min = np.min(L_onda)
Fv_ordenado = np.sort(Fv)
muestra = np.linspace(L_onda_min,
                    L_onda_max, np.shape(L_onda)[0])
Fv_modelo_ordenado = np.sort(perfil_ajuste_gauss(muestra, Optimos[0],Optimos[1],Optimos[2],
                    Optimos[3],Optimos[4]))

D, p_value = scipy.stats.kstest(Fv_ordenado, func_distribucion, args=(Fv_modelo_ordenado,))

print "D gauss, p-value Gauss", D, p_value


# Modelo gaussiano

Fv_modelo_ordenado = np.sort(perfil_ajuste_lorentz(muestra, Optimos2[0],Optimos2[1],Optimos2[2],
                    Optimos2[3],Optimos2[4]))

D, p_value = scipy.stats.kstest(Fv_ordenado, func_distribucion, args=(Fv_modelo_ordenado,))

print "D Lorentz, p-value Lorentz", D, p_value
