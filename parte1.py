'''
Este script encuentra fits optimos para un set de datos experimentales
usando dos tipos de modelos: Gaussiano y de Lorentz, con una adivinanza
inicial para los parametros que los definen.
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy import optimize as opt

def recta(params, x):
    a = params[0]
    b = params[1]
    return a * x + b


def gaussiana(params, x):
    A = params[0]
    mu = params[1]
    sigma = params[2]
    y = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y

def lorentz(params, x):
    A = params[0]
    mu = params[1]
    sigma = params[2]
    y = A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return y


def func_modelo_gauss(params, x):
    a, b, A, mu, sigma = params
    params_recta = a, b
    params_gauss = A, mu, sigma
    rect = recta(params_recta, x)
    gauss = gaussiana(params_gauss, x)
    modelo = rect - gauss
    return modelo


def func_modelo_lorentz(params,x):
    a, b, A, mu, sigma = params
    params_recta = a, b
    params_lorentz = A, mu, sigma
    rect = recta(params_recta, x)
    lor = lorentz(params_lorentz, x)
    modelo = rect - lor
    return modelo

def func_a_minimizar_gauss(x, a, b, A, mu, sigma):
    params = a, b, A, mu, sigma
    return func_modelo_gauss(params, x)


def func_a_minimizar_lorentz(x, a, b, A, mu, sigma):
    params = a, b, A, mu, sigma
    return func_modelo_lorentz(params, x)

def chi_cuadrado(x_data, y_data, modelo, params):
    x_modelo = x_data
    y_modelo = modelo(params, x_modelo)
    chi_cuad = (y_data - y_modelo) ** 2
    Chi_cuad = np.sum(chi_cuad)
    return Chi_cuad


# Main

# Leer el archivo, datos experimentales
datos = np.loadtxt("espectro.dat")
w_length = datos[:,0]
fnu = datos[:,1]

# Setup

# Adivinanza parametros. Vector de la forma (a, b, A, mu, sigma)
# Recta a * x + b = y ; Gauss y Lorentz A amplitud, mu centro, sigma varianza.
a0 = 0, 1.39e-16, 0.1e-16, 6560, 10

# Calcula parametros optimos. Gauss y Lorentz
resultado_gauss = curve_fit(func_a_minimizar_gauss, w_length, fnu, a0)
print "Parametros (a,b,A,mu,sigma) Gauss: ", resultado_gauss[0]
params_opt_gauss = resultado_gauss[0]

resultado_lorentz = curve_fit(func_a_minimizar_lorentz, w_length, fnu, a0)
print "Parametros (a,b,A,mu,sigma) Lorentz: ", resultado_lorentz[0]
params_opt_lorentz = resultado_lorentz[0]

# Chi cuadrado
chi_gauss = chi_cuadrado(w_length, fnu, func_modelo_gauss, a0)
chi_lorentz = chi_cuadrado(w_length, fnu, func_modelo_lorentz, a0)
print "Chi cuadrado (Gauss):", chi_gauss
print "Chi cuadrado (Lorentz):", chi_lorentz

# Plot Gauss, Lorentz y datos experimentales
y_optimo_gauss = func_modelo_gauss(params_opt_gauss, w_length)
y_optimo_lorentz = func_modelo_lorentz(params_opt_lorentz, w_length)

fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)

ax1.plot(w_length, fnu, '*', label="Datos Experimentales")
ax1.plot(w_length, y_optimo_lorentz, label="Ajuste Lorentz")
ax1.plot(w_length, y_optimo_gauss, label="Ajuste Gauss")
ax1.set_xlabel("Longitud de onda $[Angstrom]$")
ax1.set_ylabel("Flujo por unidad de frecuencia $[erg / s / Hz / cm^2]$")
ax1.set_title("Grafico de flujo versus longitud de onda")

plt.legend(loc='lower right')
plt.draw()
plt.show()
