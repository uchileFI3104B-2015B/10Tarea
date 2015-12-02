from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy import optimize as opt
from scipy.optimize import (leastsq, curve_fit)
from scipy.stats import kstwobign
from scipy.stats import kstest

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
print "Parametros (M,N,amplitud,centro,varianza) Lorentz: ", resultado_Lorentz[0]
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








'''
Ahora implementamos el resto del codigo necesario apra la pregunta 2, usamos dos funciones más
'''
def cdf(data, modelo):
    return np.array([np.sum(modelo <= yy) for yy in data]) / len(model)

def distribucion_acumulada(x, y, parametros_optimos, Resta_modelo):
    '''
    ordena los datos
    '''
    xmin = np.min(longitud_Onda)
    xmax = np.max(longitud_Onda)
    y_data_sorted = np.sort(radiacion)
    y_model_sorted= np.sort(Resta_modelo(parametros_optimos, np.linspace(xmin, xmax, 1000)))
    return y_model_sorted, y_data_sorted





# Kolmogorov-Smirnov Test
y_model_sorted_Gauss, y_data_sorted = distribucion_acumulada(longitud_Onda, radiacion,
                                                          parametros_optimos_Gauss,
                                                          Resta_modelo_Gauss)

y_model_sorted_Lorentz, y_data_sorted = distribucion_acumulada(longitud_Onda, radiacion,
                                                          parametros_optimos_Lorentz,
                                                          Resta_modelo_Lorentz)
N = len(y_data_sorted)

CDF_model_Gauss = cdf(y_data_sorted, y_model_sorted_Gauss)
CDF_model_Lorentz = cdf(y_data_sorted, y_model_sorted_Lorentz)

# Graficos de probabilidad acumulada
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(y_data_sorted, np.arange(N) / N, '-^',
         drawstyle='steps-post', label="Datos experimentales")
ax1.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
ax1.plot(y_data_sorted, CDF_model_Gauss, '-x',
         drawstyle='steps-post', label="Modelo Gauss")
ax1.set_title('Probabilidad acumulada')
ax1.set_xlabel('Longitud de onda $[Angstrom]$')
ax1.set_ylabel('Probabilidad')
ax1.legend(loc='upper left')

ax2.plot(y_data_sorted, np.arange(N) / N, '-^',
         drawstyle='steps-post', label="Datos experimentales")
ax2.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
ax2.plot(y_data_sorted, CDF_model_Lorentz, 'g-x',
         drawstyle='steps-post', label="Modelo Lorentz")
ax2.set_title('Probabilidad acumulada')
ax2.set_xlabel('Longitud de onda $[Angstrom]$')
ax2.set_ylabel('Probabilidad')
ax2.legend(loc='upper left')
plt.tight_layout()
plt.draw()
plt.show()




#Ahora vemos cual es más acertado en base a la distancia con los puntos de la muestra, diremos que la distancia critica será Dn_critico

Dn_scipy_Gauss, prob_scipy_Gauss = kstest(y_data_sorted,
                                  cdf, args=(y_model_sorted_Gauss,))
print "Dn_scipy Gauss  : ", Dn_scipy_Gauss
print "Nivel de confianza (Gauss) : ", prob_scipy_Gauss

Dn_scipy_Lorentz, prob_scipy_Lorentz = kstest(y_data_sorted,
                                  cdf, args=(y_model_sorted_Lorentz,))
print "Dn_scipy Lorentz : ", Dn_scipy_Lorentz
print "Nivel de confianza (Lorentz) : ", prob_scipy_Lorentz

ks_dist = kstwobign() # two-sided, aproximacion para big N
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
print "Dn_critico = ", Dn_critico
