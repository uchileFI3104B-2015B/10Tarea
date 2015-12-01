'''
Este script
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


def func_modelo(params, x):
    a, b, A, mu, sigma = params
    params_recta = a, b
    params_gauss = A, mu, sigma
    rect = recta(params_recta, x)
    gauss = gaussiana(params_gauss, x)
    modelo = rect - gauss
    return modelo


def func_a_minimizar_gauss(x, a, b, A, mu, sigma):
    params = a, b, A, mu, sigma
    return func_modelo(params, x)


# Main
# Leer el archivo, datos experimentales
datos = np.loadtxt("espectro.dat")
w_length = datos[:,0]
fnu = datos[:,1]
# Setup
# Adivinanza parametros. Vector de la forma (a, b, A, mu, sigma)
# Recta a * x + b = y ; Gauss A amplitud, mu centro, sigma varianza.
a0 = 0, 1.39e-16, 0.1e-16, 6560, 10
resultado_gauss = curve_fit(func_a_minimizar_gauss, w_length, fnu, a0)
print "Parametros (a,b,A,mu,sigma): ", resultado_gauss[0]
params_opt = resultado_gauss[0]

# Plot Gauss y datos experimentales
y_optimo = func_modelo(params_opt, w_length)

fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)
ax1.plot(w_length, fnu, '*', label="Datos Experimentales")
ax1.plot(w_length, y_optimo, label="Ajuste Gaussiano-Recta")
ax1.set_xlabel("Fnu")
ax1.set_ylabel("Wavelength")
plt.legend(loc='lower right')
plt.draw()
plt.show()
