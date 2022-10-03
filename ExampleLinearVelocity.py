# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:28:18 2022

@author: kriog
"""

import IceWaveTools2 as iwt 
import numpy  as np
import matplotlib.pyplot as plt

"""This is an example code m is the number of grid points and l is the radious of the domain, the domain is square formed"""
l,m = 250, 500 
 

"""The defining fetures of the LVinEnd is that it calculates the end point"""
t = 200
x_end = np.array([0,0])
v = np.array([10,0])

"""op(l,m) return a dictionary with all of the opreators including the X,Y vectors and the forurier spectral vectors ksi1,ksi2"""

op = iwt.GetOperators(l, m)

X,Y = op['X'],op['Y']


I = iwt.LVinEnd(x_end, v, t, l, m)

"""EtaofI(I,n=1) returns the """
eta = iwt.EtaofI(I,n = 1)


plt.contourf(X,Y,eta)
