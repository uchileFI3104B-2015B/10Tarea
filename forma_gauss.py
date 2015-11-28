#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que aproxima el espectro por la suma de una recta y una funcion
Gaussiana usando Levenberg- Marquardt'''

def importar_datos(txt):
    ''' función que retorna arreglos por cada columna del archivo de datos'''
    data = np.loadtxt(txt)
    return data[:, 0], data[:, 1]


def mostrar_datos(data, xlabel, ylabel, title, ylim):
    ''' función para graficar los datos'''
    ax, fig = plt.subplots()
    plt.plot(data[0], data[1])
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_ylim(ylim)
    plt.savefig("{}.jpg".format(title))


# main
wlength, fnu = importar_datos("espectro.dat")
mostrar_datos([wlength, fnu], "Wavelength",
              "$F_\\nu [erg s^{-1} Hz^{-1} cm^{-2}]$",
              "espectro", [1.28e-16, 1.42e-16])
plt.show()
