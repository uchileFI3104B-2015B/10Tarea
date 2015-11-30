#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
este codigo busca modelar un espectro experimental
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy import optimize as opt

# funciones estructurales


def leer_archivo(nombre):
    '''
    lee el archivo
    nombre debe ser un str
    '''
    datos = np.loadtxt(nombre)
    return datos


# inicializacion
print leer_archivo("espectro.dat")
