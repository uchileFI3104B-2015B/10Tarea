import numpy as n
import matplotlib.pyplot as p
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import kstest


# Funciones
def Gauss(x, Parametros):
    A, u, s, a, b = Parametros  # Amplitud, centro, varianza, alfa y beta
    return -A * norm(loc=u, scale=s).pdf(x) + a*x + b


def Lorentz(x, Parametros):
    A, u, s, a, b = Parametros
    return -A * cauchy(loc=u, scale=s).pdf(x) + a*x + b


def Derivada(x, Modelo, Indice, Parametros):
    h = 10**(-9)  # h tendiendo a cero y aplica definicion de derivada
    hParcial = n.zeros(5)
    hParcial[Indice] = h  # Deriva con respecto a uno de los parametros
    return (Modelo(x, Parametros) - Modelo(x, Parametros - hParcial))/h


def X2(Datos, Modelo, Parametros):
    return n.sum(n.power((Datos[1] - Modelo(Datos[0], Parametros)), 2))


def alfa(Datos, Modelo, Parametros, Lambda):
    Derivadas = n.zeros((5, len(Datos[0])))
    aKL = n.zeros((5, 5))
    for Indice in range(5):
        Derivadas[Indice] = Derivada(Datos[0], Modelo, Indice, Parametros)
    for k in range(5):
        for L in range(5):
            aKL[k][L] = n.sum(Derivadas[k]*Derivadas[L])
    for j in range(5):
        aKL[j][j] = aKL[j][j] * (1 + Lambda)
    return aKL


def beta(Datos, Modelo, Parametros):
    bK = n.zeros(5)
    F1 = Datos[1] - Modelo(Datos[0], Parametros)
    for Indice in range(5):
        F2 = Derivada(Datos[0], Modelo, Indice, Parametros)
        bK[Indice] = n.sum(F1*F2)
    return bK


def cdf(Datos, Modelo):
    return n.array([n.sum(Modelo <= a) for a in Datos]) / len(Modelo)

# Datos iniciales
LongitudOnda = n.loadtxt("espectro.dat", usecols=(0,))
Frecuencia = n.loadtxt("espectro.dat", usecols=(1,))
Datos = [[], []]
for a in range(len(LongitudOnda)):
    Datos[0].append(LongitudOnda[a])
    Datos[1].append((10**16)*Frecuencia[a])
Datos[0] = n.asarray(Datos[0])
Datos[1] = n.asarray(Datos[1])
Fit = n.polyfit(Datos[0], Datos[1], 1)  # Obtiene los coeficientes de la recta
# Modelo de Gauss
ParametrosGauss = (1, 6565, 5, Fit[0], Fit[1])
X2Gauss = X2(Datos, Gauss, ParametrosGauss)
Lambda = 0.001  # Lambda y deltaX iniciales que se van ajustando
deltaX = 10
while deltaX > 0.00001:  # Obtiene la normal que mejor se ajusta comparando
    a = alfa(Datos, Gauss, ParametrosGauss, Lambda)
    b = beta(Datos, Gauss, ParametrosGauss)
    deltaParametros = n.dot(n.linalg.inv(a), b)
    X2t = X2(Datos, Gauss, ParametrosGauss + deltaParametros)
    if X2t >= X2Gauss:
        Lambda = Lambda * 10
    else:
        Lambda = Lambda * 0.1
        ParametrosGauss = ParametrosGauss + deltaParametros
        deltaX = X2Gauss - X2t
        X2Gauss = X2t
# Modelo de Lorentz
ParametrosLorentz = (1, 6565, 5, Fit[0], Fit[1])
X2Lorentz = X2(Datos, Lorentz, ParametrosLorentz)
Lambda = 0.001
deltaX = 10
while deltaX > 0.00001:  # Obtiene la cauchy que mejor se ajusta comparando
    a = alfa(Datos, Lorentz, ParametrosLorentz, Lambda)
    b = beta(Datos, Lorentz, ParametrosLorentz)
    deltaParametros = n.dot(n.linalg.inv(a), b)
    X2t = X2(Datos, Lorentz, ParametrosLorentz + deltaParametros)
    if X2t >= X2Lorentz:
        Lambda = Lambda * 10
    else:
        Lambda = Lambda * 0.1
        ParametrosLorentz = ParametrosLorentz + deltaParametros
        deltaX = X2Lorentz - X2t
        X2Lorentz = X2t
# Kolmogorov-Smirnov
xMin = n.min(Datos[0])
xMax = n.max(Datos[0])
x = n.linspace(xMin, xMax, 10000)
yGauss = n.sort(Gauss(x, ParametrosGauss)) / 10**16
yLorentz = n.sort(Lorentz(x, ParametrosLorentz)) / 10**16
y = n.sort(Datos[1]) / 10**16
cdfGauss = cdf(y, yGauss)
cdfLorentz = cdf(y, yLorentz)
N = len(y)
# Aplica el testy obtiene los valores
DnGauss, ConfianzaGauss = kstest(y, cdf, args=(yGauss,))
DnLorentz, ConfianzaLorentz = kstest(y, cdf, args=(yLorentz,))
# Resultados
print ('Gauss:')
print ('a = '+str(round(ParametrosGauss[3], 8))+' 10^-16')
print ('b = '+str(round(ParametrosGauss[4], 4))+' 10^-16')
print ('A = '+str(round(ParametrosGauss[0], 4))+' 10^-16 ')
print ('u = '+str(round(ParametrosGauss[1], 4)))
print ('s = '+str(round(ParametrosGauss[2], 4)))
print ('X^2 = '+str(round(X2Gauss, 6))+' 10^-32 ')
print ('Dn = '+str(round(DnGauss, 4)))
print ('Nivel de confianza: '+str(round(ConfianzaGauss, 5)))
print ('Lorentz:')
print ('a = '+str(round(ParametrosLorentz[3], 8))+' 10^-16')
print ('b = '+str(round(ParametrosLorentz[4], 4))+' 10^-16')
print ('A = '+str(round(ParametrosLorentz[0], 4))+' 10^-16 ')
print ('u = '+str(round(ParametrosLorentz[1], 4)))
print ('s = '+str(round(ParametrosLorentz[2], 4)))
print ('X^2 = '+str(round(X2Lorentz, 6))+' 10^-32 ')
print ('Dn = '+str(round(DnLorentz, 4)))
print ('Nivel de confianza: '+str(round(ConfianzaLorentz, 5)))
# Plots
fig = p.figure()
p.plot(Datos[0], Datos[1] / 10**16, 'g', label='$Espectro$ $observado$')
p.plot(x, Gauss(x, ParametrosGauss) / 10**16, 'r', label='$Perfil$ $Gauss$')
p.plot(x, Lorentz(x, ParametrosLorentz) / 10**16, label='$Perfil$ $Lorentz$')
p.xlabel('$Longitud$ $de$ $onda$ $[\AA]}$')
p.ylabel('$F_{\\nu} \, [erg \, s^{-1}Hz^{-1}cm^{-2}]$')
p.title('$Espectro$ $y$ $Fits$')
p.legend(loc='lower left')
p.xlim([6460, 6660])
fig.savefig('Fits.png')
p.show()
fig = p.figure()
p.plot(y, n.arange(N) / N, 'g', drawstyle='steps-post',
       label='$CDF$ $Observada$')
p.plot(y, cdfGauss, 'r', drawstyle='steps-post',
       label='$CDF$ $Modelo$ $de$ $Gauss$')
p.plot(y, cdfLorentz, 'b', drawstyle='steps-post',
       label='$CDF$ $Modelo$ $de$ $Lorentz$')
p.xlabel('$F_{\\nu} \, [erg \, s^{-1}Hz^{-1}cm^{-2}]$')
p.title('$Funcion$ $de$ $Distribucion$ $acumulada$')
p.legend(loc='upper left')
fig.savefig('Distribucion.png')
p.show()
