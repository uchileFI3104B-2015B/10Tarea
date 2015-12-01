'''
Este script
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy.stats import kstwobign
from scipy.stats import kstest


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

def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)

def distribucion_acumulada(x, y, params_opt, func_modelo):
    xmin = np.min(w_length)
    xmax = np.max(w_length)
    y_data_sorted = np.sort(fnu)
    y_model_sorted= np.sort(func_modelo(params_opt, np.linspace(xmin, xmax, 1000)))
    return y_model_sorted, y_data_sorted


# Main

# Leer el archivo, datos experimentales
datos = np.loadtxt("espectro.dat")
w_length = datos[:,0]
fnu = datos[:,1]

# Setup

# Adivinanza parametros. Vector de la forma (a, b, A, mu, sigma)
# Recta a * x + b = y ; Gauss A amplitud, mu centro, sigma varianza.
a0 = 0, 1.39e-16, 0.1e-16, 6560, 10

# Calcula parametros optimos. Gauss y Lorentz
resultado_gauss = curve_fit(func_a_minimizar_gauss, w_length, fnu, a0)
print "Parametros (a,b,A,mu,sigma) Gauss: ", resultado_gauss[0]
params_opt_gauss = resultado_gauss[0]

resultado_lorentz = curve_fit(func_a_minimizar_lorentz, w_length, fnu, a0)
print "Parametros (a,b,A,mu,sigma) Lorentz: ", resultado_lorentz[0]
params_opt_lorentz = resultado_lorentz[0]

# Kolmogorov-Smirnov Test
y_model_sorted_g, y_data_sorted = distribucion_acumulada(w_length, fnu,
                                                          params_opt_gauss,
                                                          func_modelo_gauss)

y_model_sorted_l, y_data_sorted = distribucion_acumulada(w_length, fnu,
                                                          params_opt_lorentz,
                                                          func_modelo_lorentz)
N = len(y_data_sorted)
CDF_model_g = cdf(y_data_sorted, y_model_sorted_g)
CDF_model_l = cdf(y_data_sorted, y_model_sorted_l)

# Graficos de probabilidad acumulada
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(y_data_sorted, np.arange(N) / N, '-^',
         drawstyle='steps-post', label="Datos experimentales")
ax1.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
ax1.plot(y_data_sorted, CDF_model_g, '-x',
         drawstyle='steps-post', label="Modelo Gauss")
ax1.set_title('Probabilidad acumulada')
ax1.set_xlabel('Longitud de onda $[Angstrom]$')
ax1.set_ylabel('Probabilidad')
ax1.legend(loc='upper left')

ax2.plot(y_data_sorted, np.arange(N) / N, '-^',
         drawstyle='steps-post', label="Datos experimentales")
ax2.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
ax2.plot(y_data_sorted, CDF_model_l, 'g-x',
         drawstyle='steps-post', label="Modelo Lorentz")
ax2.set_title('Probabilidad acumulada')
ax2.set_xlabel('Longitud de onda $[Angstrom]$')
ax2.set_ylabel('Probabilidad')
ax2.legend(loc='upper left')
plt.tight_layout()
plt.draw()
plt.show()

# Usando kstest
Dn_scipy_g, prob_scipy_g = kstest(y_data_sorted,
                                  cdf, args=(y_model_sorted_g,))
print "Dn_scipy Gauss  : ", Dn_scipy_g
print "Nivel de confianza (Gauss) : ", prob_scipy_g

Dn_scipy_l, prob_scipy_l = kstest(y_data_sorted,
                                  cdf, args=(y_model_sorted_l,))
print "Dn_scipy Lorentz : ", Dn_scipy_l
print "Nivel de confianza (Lorentz) : ", prob_scipy_l

ks_dist = kstwobign() # two-sided, aproximacion para big N
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
print "Dn_critico = ", Dn_critico
