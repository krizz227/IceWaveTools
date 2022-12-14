# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:55:01 2022

@author: kriog
"""

import IceWave as iwt 
import numpy  as np
import matplotlib.pyplot as plt

"""This is an example code m is the number of grid points and l is the radious of the domain, the domain is square formed"""
l,m = 150, 300 
 

"""The defining fetures of the LVinEnd is that it calculates the end point"""
t = 200
x_end = np.array([0,0])
v = np.array([10,0])

"""op(l,m) return a dictionary with all of the opreators including the X,Y vectors and the forurier spectral vectors ksi1,ksi2"""

op = iwt.GetOperators(l, m)

X,Y = op['X'],op['Y']

v = 10

r = 30

t_end = 1

alpha = v/r


#acceleration
a = 0

a = a/r

n = 2

T = np.linspace(0,t_end,n)

#The functions X_1,X_2 is the parametreisation of the path and can be changed

def X_1(tau):
    return  r*np.cos((alpha + 0.5*a*tau)*tau) 


def X_2(tau):
    return r*np.sin((alpha + 0.5*a*tau)*tau)

print("warning this does take a loong time, the method is not optimized")

I = iwt.GP(X_1, X_2, l, m, T)

"""EtaofI(I,n=2) returns the """
eta = iwt.EtaofI(I,n = 2)


plt.contourf(X,Y,eta[-1])
