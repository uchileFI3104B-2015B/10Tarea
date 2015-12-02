'''
Este script encuentra
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy.stats import kstwobign


# Creamos funciones
def recta(params, x):
    # Funcion de la recta
    a = params[0]
    b = params[1]
    return a * x + b


def gauss(params, x):
    # Funcion de gauss
    A = params[0]
    mu = params[1]
    sigma = params[2]
    return A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def lorentz(params, x):
    # Funcion de Lorentz
    A = params[0]
    mu = params[1]
    sigma = params[2]
    return A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def f_modelo_g(params, x):
    # Funcion modelo para Gauss
    a, b, A, mu, sigma = params
    params_f = a, b
    params_g = A, mu, sigma
    f_recta = recta(params_f, x)
    f_gauss = gauss(params_g, x)
    return f_recta - f_gauss


def f_modelo_l(params, x):
    # Funcion modelo para Lorentz
    a, b, A, mu, sigma = params
    params_f = a, b
    params_l = A, mu, sigma
    f_recta = recta(params_f, x)
    f_lorentz = lorentz(params_l, x)
    return f_recta - f_lorentz


def f_minimizar_g(x, a, b, A, mu, sigma):
    # Funcion a minimzar con curve_fit para Gauss
    params = a, b, A, mu, sigma
    return f_modelo_g(params, x)


def f_minimizar_l(x, a, b, A, mu, sigma):
    # Funcion a minimizar con curve_fit para Gauss
    params = a, b, A, mu, sigma
    return f_modelo_l(params, x)


# Main
# Setup
# Cargamos los datos
datos = np.loadtxt("espectro.dat")
wave_length = datos[:, 0]
f_nu = datos[:, 1]

# Creamos adivinanzas
ad_a = 0
ad_b = 1.4e-16
ad_A = 0.1e-16
ad_mu = 6580
ad_sigma = 10

adivinanzas = ad_a, ad_b, ad_A, ad_mu, ad_sigma

# Realizamos arreglo usando curve_fit para gauss y lorentz respectivamente
params_g, cov_g = curve_fit(f_minimizar_g, wave_length, f_nu, adivinanzas)
params_l, cov_l = curve_fit(f_minimizar_l, wave_length, f_nu, adivinanzas)

# Guradamos los parametros obtenidos en nuevas variables
a_g = params_g[0]
b_g = params_g[1]
A_g = params_g[2]
mu_g = params_g[3]
sigma_g = params_g[4]

a_l = params_l[0]
b_l = params_l[1]
A_l = params_l[2]
mu_l = params_l[3]
sigma_l = params_l[4]

# Mostramos los parámetros
print "-------------------------------------------------------"
print "Gauss:"
print "Parametros a, b:", a_g, ", ", b_g
print "Parametros A, mu, sigma:", A_g, ", ", mu_g, ", ", sigma_g
print "-------------------------------------------------------"
print "Lorentz"
print "Parametros a, b:", a_l, ", ", b_l
print "Parametros A, mu, sigma:", A_l, ", ", mu_l, ", ", sigma_l


# P2
# Renombramos los vectores de parametros
r_g = params_g
r_l = params_l

# Construccion de la funcion de probabilidad acumulada para el KS test
# para Gauss y Lorentz
x_min = np.min(wave_length)
x_max = np.max(wave_length)

f_modelo_g_sorted = np.sort(f_modelo_g(r_g, (np.linspace(x_min, x_max, 1000))))
f_modelo_l_sorted = np.sort(f_modelo_l(r_l, (np.linspace(x_min, x_max, 1000))))

f_nu_sorted = np.sort(f_nu)

CDF_model_g = (np.array([np.sum(f_modelo_g_sorted <= yy)
               for yy in f_nu_sorted]) / len(f_modelo_g_sorted))

CDF_model_l = (np.array([np.sum(f_modelo_l_sorted <= yy)
               for yy in f_nu_sorted]) / len(f_modelo_l_sorted))

N = len(wave_length)

max_1_g = np.max(CDF_model_g - np.arange(N) / N)
max_1_l = np.max(CDF_model_l - np.arange(N) / N)

max_2_g = np.max(np.arange(1, N+1)/N - CDF_model_g)
max_2_l = np.max(np.arange(1, N+1)/N - CDF_model_l)

# Obtenemos nuestra mayor distancia absoluta para Gauss y Lorentz
Dn_g = max(max_1_g, max_2_g)
Dn_l = max(max_1_l, max_2_l)

# Mostramos los Dn para Gauss y Lorentz
print "Dn para gauss: ", Dn_g
print "Dn para lorentz: ", Dn_l

# Obtenemos Dn critico
ks_dist = kstwobign()
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
print "Dn critico: ", Dn_critico

# Graficamos
y_optimo_g = f_modelo_g(params_g, wave_length)
y_optimo_l = f_modelo_l(params_l, wave_length)

# Para P1
fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)
ax1.plot(wave_length, f_nu, '*', label="Datos Experimentales")
ax1.plot(wave_length, y_optimo_g, label="Ajuste Gaussiano-Recta")
ax1.plot(wave_length, y_optimo_l, label="Ajuste Lorentz-Recta")
ax1.set_title("Grafico de flujo vs longitud de onda")
ax1.set_xlabel("Longitud de onda $[Angstrom]$")
ax1.set_ylabel("Flujo por unidad de frecuencia [$erg/s/Hz/cm^2}$]")
plt.legend(loc=4)

# Para P2
fig, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)

ax2.plot(f_nu_sorted, np.arange(N) / N, '-^',
         drawstyle='steps-post', label="Datos experimentales")
ax2.plot(f_nu_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
ax2.plot(f_nu_sorted, CDF_model_g, '-x',
         drawstyle='steps-post', label="Modelo Gauss")
ax2.set_title('Probabilidad acumulada')
ax2.set_xlabel('Longitud de onda $[Angstrom]$')
ax2.set_ylabel('Probabilidad')
ax2.legend(loc='upper left')

ax3.plot(f_nu_sorted, np.arange(N) / N, '-^',
         drawstyle='steps-post', label="Datos experimentales")
ax3.plot(f_nu_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
ax3.plot(f_nu_sorted, CDF_model_l, 'g-x',
         drawstyle='steps-post', label="Modelo Lorentz")
ax3.set_title('Probabilidad acumulada')
ax3.set_xlabel('Longitud de onda $[Angstrom]$')
ax3.set_ylabel('Probabilidad')
ax3.legend(loc='upper left')
plt.tight_layout()
plt.draw()
plt.show()
