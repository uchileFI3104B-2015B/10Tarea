from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit


def gauss(x, A, mu, sigma, a, b):
    '''
    Funcion lineal menos una funcion gaussiana evaluada en un x dado.
    '''
    return (a * x + b) - (A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x))


def lorentz(x, A, mu, sigma, a, b):
    '''
    Funcion lineal menos un perfil de Lorentz evaluada en un x dado.
    '''
    return (a * x + b) - (A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x))


def chi_2(wave, fnu, modelo, param):
    '''
    Calcula chi^2 para uno de los modelos a partir de los datos
    '''
    y_modelo = modelo(wave, param[0], param[1], param[2], param[3], param[4])
    chi_2 = (fnu - y_modelo) ** 2
    return np.sum(chi_2)


def cdf(datos, modelo):
    return np.array([np.sum(modelo <= yy) for yy in datos]) / (1.0*len(modelo))


#Main
data = np.loadtxt("espectro.dat")
wave = data[:, 0]
fnu = data[:, 1]


#El ajuste
opt_lorentz, pcov = curve_fit(lorentz, wave, fnu, p0=[0.1e-16, 6550, 10, 0.04e-18, 1])
opt_gauss, pcov = curve_fit(gauss, wave, fnu, p0=[0.1e-16, 6550, 10, 0.04e-18, 1])


#Las funciones que optimizan
l = np.linspace(6.4600e+03, 6.6596e+03, 200)
y_lorentz = lorentz(l, opt_lorentz[0], opt_lorentz[1], opt_lorentz[2], opt_lorentz[3], opt_lorentz[4])
y_gauss = gauss(l, opt_gauss[0], opt_gauss[1], opt_gauss[2], opt_gauss[3], opt_gauss[4])


#El chi^2
chi_lorentz = chi_2(wave, fnu, lorentz, opt_lorentz)
chi_gauss = chi_2(wave, fnu, gauss, opt_gauss)

print "Ajuste Lorentziano"
print "Pendiente = {}, coef. de posicion = {}, Amplitud = {}, Centro = {} y Varianza = {}".format(
            opt_lorentz[3], opt_lorentz[4], opt_lorentz[0], opt_lorentz[1], opt_lorentz[2])

print "Ajuste Gaussiano"
print "Pendiente = {}, coef. de posicion = {}, Amplitud = {}, Centro = {} y Varianza = {}".format(
            opt_gauss[3], opt_gauss[4], opt_gauss[0], opt_gauss[1], opt_gauss[2])

print "---------------------------------------------------------------------------------------------"


print "chi^2 para el ajuste Lorentziano", chi_lorentz
print "chi^2 para el ajuste Gaussiano", chi_gauss

print "---------------------------------------------------------------------------------------------"

#El plot
fig = plt.figure()
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.scatter(wave, fnu, label="Espectro")
plt.plot(l, y_lorentz, 'r', label='Perfil de Lorentz')
plt.plot(l, y_gauss, 'g-', label='Perfil Gaussiano')
plt.legend(loc=4)
plt.xlabel("Wavelength[$\AA$]")
plt.ylabel("$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$")
plt.show()


#K-S test
xmin = np.min(wave)
xmax = np.max(wave)
y_model_sorted_l = np.sort(lorentz(np.linspace(xmin, xmax, 1000), opt_lorentz[0], opt_lorentz[1],
                                 opt_lorentz[2], opt_lorentz[3], opt_lorentz[4]))

y_model_sorted_g = np.sort(lorentz(np.linspace(xmin, xmax, 1000), opt_gauss[0], opt_gauss[1],
                                 opt_gauss[2], opt_gauss[3], opt_gauss[4]))

y_data_sorted = np.sort(fnu)

CDF_model_l = np.array([np.sum(y_model_sorted_l <= yy) for yy in y_data_sorted]) / len(y_model_sorted_l)
CDF_model_g = np.array([np.sum(y_model_sorted_g <= yy) for yy in y_data_sorted]) / len(y_model_sorted_g)

N = len(y_data_sorted)

fig = plt.figure()
plt.plot(y_data_sorted, np.arange(N) / N, '-^', drawstyle='steps-post', label="CDF inferior obs")
plt.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post', label="CDF superior obs")
plt.plot(y_data_sorted, CDF_model_l, '-x', drawstyle='steps-post', label="CDF modelo Lorentziano")
plt.xlabel("$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$")
plt.legend(loc=2)
plt.show()

fig = plt.figure()
plt.plot(y_data_sorted, np.arange(N) / N, '-^', drawstyle='steps-post', label="CDF inferior obs")
plt.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post', label="CDF superior obs")
plt.plot(y_data_sorted, CDF_model_g, '-o', drawstyle='steps-post', label="CDF modelo Gaussiano")
plt.xlabel("$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$")
plt.legend(loc=2)
plt.show()

max1_l = np.max(CDF_model_l - np.arange(N) / N)
max2_l = np.max(np.arange(1, N+1)/N - CDF_model_l)

Dn_l = max(max1_l, max2_l)
print "Dn para ajuste lorentziano = ", Dn_l

max1_g = np.max(CDF_model_g - np.arange(N) / N)
max2_g = np.max(np.arange(1, N+1)/N - CDF_model_l)

Dn_g = max(max1_g, max2_g)
print "Dn para ajuste lorentziano = ", Dn_g

print "---------------------------------------------------------------------------------------------"

Dn_scipy_l, prob_scipy_l = scipy.stats.kstest(y_data_sorted, cdf, args=(y_model_sorted_l,))

Dn_scipy_g, prob_scipy_g = scipy.stats.kstest(y_data_sorted, cdf, args=(y_model_sorted_g,))

print "Nivel de confianza para ajuste lorentziano : ", prob_scipy_l

print "Nivel de confianza para ajuste gaussiano : ", prob_scipy_g

