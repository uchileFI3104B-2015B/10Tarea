from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy import optimize as opt
from scipy.optimize import (leastsq, curve_fit)

'''
codigo XD
'''

'''
Recordar cambiar los argumentos de las funciones para que tomen X y una lista con los parametros
'''

def Recta(parametros, x):
    M = parametros[0]
    N = parametros[1]
    y = x*M + N
    return y

def Gaussiana(parametros, x):
    amplitud = parametros[0]
    centro = parametros[1]
    varianza = parametros[2]
    g = amplitud * scipy.stats.norm(loc=centro, scale=varianza).pdf(x)
    return g

def Lorentz(parametros, x):
    amplitud, centro, varianza = parametros
    return amplitud * scipy.stats.cauchy(loc=centro, scale=varianza).pdf(x)

def Resta_modelo_Gauss(parametros, elementosX):
    '''
    "parametros" debe ser una tupla o lista con todos los parametros de "Gaussiana" y "Recta"
    Devuelve un vector con todos los varialos "y" para la ecuacion "Recta" - "Gaussiana"
    '''
    M, N, amplitud, centro, varianza = parametros
    parametros_Recta = M, N
    parametros_Gauss = amplitud, centro, varianza
    Rec = Recta(parametros_Recta, elementosX)
    Gau = Gaussiana(parametros_Gauss, elementosX)
    resta = Rec - Gau
    return resta

def Resta_modelo_Lorentz(parametros, elementosX):
    '''
    analogo a "Resta_modelo_Gauss()"
    '''
    M, N, amplitud, centro, varianza = parametros
    parametros_Recta = M, N
    parametros_Gauss = amplitud, centro, varianza
    Rec = Recta(parametros_Recta, elementosX)
    Lor = Lorentz(parametros_Gauss, elementosX)
    resta = Rec - Lor
    return resta

def Resta_a_minimizar_Gauss(elementosX, M, N, amplitud, centro, varianza):
    parametros = M, N, amplitud, centro, varianza
    x = elementosX
    return Resta_modelo_Gauss(parametros, x)

def Resta_a_minimizar_Lorentz(elementosX, M, N, amplitud, centro, varianza):
    parametros = M, N, amplitud, centro, varianza
    x = elementosX
    return Resta_modelo_Lorentz(parametros, x)

datos = np.loadtxt("espectro.dat")
radiacion=datos[:,1]    #fν [erg s-1 Hz-1 cm-2]
longitud_Onda=datos[:,0]     #[Angstrom]


#Adivinanza inicial para el "curvefit", en el orden (M, N, amplitud, centro, varianza)
VectorAdivinanza = 0, 1.39e-16, 0.1e-16, 6560, 10
resultado_gauss = curve_fit(Resta_a_minimizar_Gauss, longitud_Onda, radiacion, VectorAdivinanza)
resultado_Lorentz = curve_fit(Resta_a_minimizar_Lorentz, longitud_Onda, radiacion, VectorAdivinanza)
print "Parametros (M,N,amplitud,centro,varianza) Gauss: ", resultado_gauss[0]   #nos devuelve una lista con los parametros a optimizar
parametros_optimos_Gauss = resultado_gauss[0]
parametros_optimos_Lorentz = resultado_Lorentz[0]



#plot
y_optimos_Gauss = Resta_modelo_Gauss(parametros_optimos_Gauss, longitud_Onda)
y_optimo_Lorentz = Resta_modelo_Lorentz(parametros_optimos_Lorentz, longitud_Onda)

fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)

ax1.plot(longitud_Onda, radiacion, '*', label="Datos Experimentales")
ax1.plot(longitud_Onda, y_optimo_Lorentz, label="Ajuste Lorentz")
ax1.plot(longitud_Onda, y_optimos_Gauss, label="Ajuste Gauss")
ax1.set_xlabel("Longitud de onda $[Angstrom]$")
ax1.set_ylabel("Flujo por unidad de frecuencia $[erg / s / Hz / cm^2]$")
ax1.set_title("Grafico de flujo versus longitud de onda")

plt.legend(loc='lower right')
plt.draw()
plt.show()
