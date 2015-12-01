'''
esteblablabla
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


def gauss(params, x):
    A = params[0]
    mu = params[1]
    sigma = params[2]
    return A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def f_modelo(params, x):
    a, b, A, mu, sigma = params
    params_f = a, b
    params_g = A, mu, sigma
    f_recta = recta(params_f, x)
    f_gauss = gauss(params_g, x)
    return f_recta -  f_gauss


def f_minimizar(a, b, A, mu, sigma, x):
    params = a, b, A, mu, sigma
    return f_modelo(params, x)


#Main
datos = np.loadtxt("espectro.dat")
wave_length = datos[:,0]
f_nu = datos[:,1]

ad_a = 0
ad_b = 8.87e-17
ad_A = 8.22e-17
ad_mu = 6563
ad_sigma = 3

adivinanzas = ad_a, ad_b, ad_A, ad_mu, ad_sigma

params, cov = curve_fit(f_minimizar, wave_length, f_nu, adivinanzas)

a = params[0]
b = params[1]
A = params[2]
mu = params[3]
sigma = params[4]

print "Parametros a, b:", a, ", ", b
print "Parametros A, mu, sigma:", A, ", ", mu, ", ", sigma

y_optimo = f_modelo(params, wave_length)

fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)
ax1.plot(wave_length, f_nu, '*', label="Datos Experimentales")
ax1.plot(wave_length, y_optimo, label="Ajuste Gaussiano-Recta")
ax1.set_xlabel("Wave length")
ax1.set_ylabel("Fnu")
plt.legend(loc=4)
plt.draw()
plt.show()
