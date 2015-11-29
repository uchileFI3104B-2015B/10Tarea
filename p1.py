import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

wavelength = np.loadtxt('espectro.dat', usecols=(0,))
fnu = np.loadtxt('espectro.dat', usecols=(1,))

a, b = np.polyfit(wavelength,fnu,1)
x = np.linspace(6450, 6700, 100)

fig = plt.figure(1)
plt.clf()
plt.plot(wavelength, fnu, 'o')
plt.plot(x, a*x + b)
plt.draw()
plt.show()
