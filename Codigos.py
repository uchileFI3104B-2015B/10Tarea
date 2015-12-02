"""
Script que realiza dos ajustes para un espectro que contiene un continuo y una
linea de absorcion. Un modelo corresponde a un perfil de Lorentz y el otro a un
perfil de Gauss. Ademas se realiza un test Kolmogorov-Smirnov a los modelos
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import (leastsq, curve_fit)


def Modelo_gauss(x, a, b, A, mu, sigma):
    """
    entrega valores de la funcion del modelo gaussiano evaluada en punto
    o set de puntos
    """
    f = (a*x + b) - A * stats.norm(loc=mu, scale=sigma).pdf(x)
    return f


def Modelo_lorentz(x, a, b, A, mu, sigma):
    """
    entrega valores de la funcion del modelo lorentziano evaluada en punto
    o set de puntos
    """
    f = (a*x + b) - A * stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return f


def ordenar(x_data, y_data, funcion, p_optimo):
    """
    Retorna arreglo ordenado de datos y de valores obtenidos con una funcion
    modelo
    """
    xmin = np.min(x_data)
    xmax = np.max(x_data)
    y_modelo_ordenado = np.sort(funcion(np.linspace(xmin, xmax, 200),
                                *p_optimo))
    y_data_ordenado = np.sort(y_data)
    return y_data_ordenado, y_modelo_ordenado


def prob_acumulada(y_data, y_modelo_ordenado):
    """
    Crea y entrega funcion de distribucion acumulada para el modelo
    """
    y_data_ordenado = np.sort(y_data)
    dist_acumulada_modelo = np.array([np.sum(y_modelo_ordenado <= yy) for yy in
                                     y_data_ordenado]) / len(y_modelo_ordenado)
    return dist_acumulada_modelo


def chi_cuadrado(y_data, y_modelo):
    """
    Retorna valor de chi cuadrado del modelo
    """
    chi2 = np.sum((y_data - y_modelo)**2)
    return chi2


# Setup
datos = np.loadtxt('espectro.dat')
longitud = datos[:, 0]
flujo = datos[:, 1]

# Parametros iniciales estimados

a, b = np.polyfit(longitud, flujo, 1)
A = 0.1e-16
mu = np.mean(longitud)
sigma = np.std(longitud)

# Busqueda de los mejores valores para los Parametros
p0 = [a, b, A, mu, sigma]
p_optimo_gauss, a_covarianza_gauss = curve_fit(Modelo_gauss, longitud, flujo,
                                               p0)
p_optimo_lorentz, a_covarianza_lorentz = curve_fit(Modelo_lorentz, longitud,
                                                   flujo, p0)

# Valores de chi cuadrado de los modelos

chi_cuadrado_gauss = chi_cuadrado(flujo, Modelo_gauss(longitud,
                                  *p_optimo_gauss))

chi_cuadrado_lorentz = chi_cuadrado(flujo, Modelo_lorentz(longitud,
                                    *p_optimo_lorentz))

print '\n Parametros modelo perfil Gauss: \n'
print 'Pendiente = ' + str(p_optimo_gauss[0])
print 'constante recta = ' + str(p_optimo_gauss[1])
print 'Amplitud = ' + str(p_optimo_gauss[2])
print 'Centro (mu) =' + str(p_optimo_gauss[3])
print 'Desviacion esetandar(sigma) = ' + str(p_optimo_gauss[4])
print '\n Parametros modelo perfil Lorentz: \n '
print 'Pendiente = ' + str(p_optimo_lorentz[0])
print 'constante recta = ' + str(p_optimo_lorentz[1])
print 'Amplitud = ' + str(p_optimo_lorentz[2])
print 'Centro (mu) =' + str(p_optimo_lorentz[3])
print 'Desviacion esetandar(sigma) = ' + str(p_optimo_lorentz[4])
print ''
print 'chi cuadrado gaussiana = ' + str(chi_cuadrado_gauss)
print 'chi cuadrado lorentziana = ' + str(chi_cuadrado_lorentz)

# Test KS

y_data_ordenado = np.sort(flujo)
CDF_modelo_gauss, y_modelo_ordenado_gauss = ordenar(longitud, flujo,
                                                    Modelo_gauss,
                                                    p_optimo_gauss)
CDF_modelo_lorentz, y_modelo_ordenado_lorentz = ordenar(longitud, flujo,
                                                        Modelo_lorentz,
                                                        p_optimo_lorentz)
Dn_gauss, prob_gauss = stats.kstest(y_data_ordenado, prob_acumulada, args=(
                                    y_modelo_ordenado_gauss,))
Dn_lorentz, prob_lorentz = stats.kstest(y_data_ordenado, prob_acumulada, args=(
                                        y_modelo_ordenado_lorentz,))
print '\n test KS \n'
print "Dn gaussiana   : ", Dn_gauss
print "Dn lorentziana   : ", Dn_lorentz
print "Nivel de confianza gaussiana : ", prob_gauss
print "Nivel de confianza lorentziana: ", prob_lorentz

# Grafico

plt.plot(longitud, flujo, color='r', label='Datos observados')
plt.plot(longitud, Modelo_gauss(longitud, *p_optimo_gauss), color='b',
         label='Modelo gaussiana')
plt.plot(longitud, Modelo_lorentz(longitud, *p_optimo_lorentz), color='g',
         label='Modelo lorentziana')
plt.title('Espectro observado y modelos perfil de gauss y lorentz \n     ')
plt.xlabel('Longitud de onda [Angstrom]')
plt.ylabel('$F_\\nu [erg\ s^{-1}Hz^{-1}cm^{-2}]$')
plt.legend(loc=4)
plt.show()
