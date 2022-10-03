# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:19:28 2022

@author: kriog
"""



from numba import jit
from scipy.optimize import newton
import numpy.fft as fft
import numpy as np
import time
from scipy.integrate import quad_vec

#-----------------------------------
#
#-----------------------------------
def Rectangle(X,Y,sides):
    m = len(X)
    
    a = sides[0]
    b = sides[1]
    Fb = np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            if np.abs(X[i,j]) <= a and np.abs(Y[i,j]) <= b:
                Fb[i,j] = 1
    return Fb


def Triangle(X,Y,sides):
    m = len(X)
    
    a = sides[0]
    b = sides[1]
    Fb = np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            if np.abs(X[i,j]) <= a and np.abs(Y[i,j]) <= b and X[i,j] >= Y[i,j]:
                Fb[i,j] = 1
    return Fb

def Gaussian(X,Y,gaus):
    return 1/gaus/np.pi*np.exp(-(X**2+Y**2)/gaus)


def Dirac(X,Y,w):
    """ here w i the scaling"""
    m = np.shape(X)
    return np.ones(m)*w


initialFuncs = {'Rectangle':Rectangle,'Triangle':Triangle,'Gaussian':Gaussian,'Dirac':Dirac}

#-----------------------------------


def CS(H,L, g = 9.81,uc = True):
    """ The critical speed given by the H and L , can obviously be modifyed with a more complicated dispersion relation"""
    HL = H/L
    
    def F(X):
        return ((3 + (2*HL/X)/np.sinh(2*HL/X))/(1-(2*HL/X)/np.sinh(2*HL/X)))**(1/4) - X
    
    X_c = newton(F,0.1)
    
    U_c = np.sqrt((X_c + 1/X_c**3)*np.tanh(HL/X_c))
    
    if uc == False:
        return U_c
    u_c = U_c*np.sqrt(g*L)
    
    
    return u_c


def GetVariables():
    
    gaus = 0.1
    gaus = 1
    
    # gravitational constant
    g = 9.81
    # The critical speed
    # variables
    rho_i = 917
    rho = 1026
    # for the dirac distribution
    epsilon = 0.01
    g = 9.81
    
    # water depth
#------------------------------------
    H = 6.8

# ------------------------------------
    # Ice thiknes
#------------------------------------
    # h = 0.01
    h = 0.17
    # h = 0.1
    # h = 0.2
    # h = 0.4
#------------------------------------
    # Elastic module
    E = 5.1*10**8
    # Flexural rigitity
    # constant for ice sigma is supposed to be 1/3
    
    sigma = 1/3
    # culd also be calculated by h and E
    
    DD = 2.35*10**5 # This is not in use
    # poisons ratio
    nu = 0.33 
    # nu= 0.33 #for the stres test


    
    #There are two alternatives for xi one is afther calculting DD (flexural rigidity the other is to calculate it through the elastic modulus)
    # xi = DD/(rho*g)
    
    xi = E*h**3/(12*rho*g*(1-sigma**2))
    
    L = xi**(1/4)
    #-----------------------------------


    B=0.41
#------------------------------------

    b = B*2*np.sqrt(rho*g*rho_i*h)
    
    v_c = CS(H,L)
    
    names =     ['gaus','g','rho_i', 'rho' ,'epsilon', 'H','L','h','E','DD','nu','b','sigma','v_c']
    Liste =     [gaus,g,rho_i,rho,epsilon,H,L,h,E,DD,nu,b,sigma,v_c]
    
    variables = dict()
    
    for (name,item) in zip(names,Liste):
        variables[name] = item
    
    globals().update(variables)
    return variables


def GetOperators(l,m,fname = 'Gaussian',arg = 'gaus'):
    
    """ Parameters : l,m Returns dx,X,Y,ksi1,ksi2,Fhatk,G_0k,Fkk,Rk,Uk """
    # fname = 'Rectangle'
    # arg = [1.5,1.5]
    variables = GetVariables()
    # fname = 'Dirac'
    # arg = 1
    
    
    rho = variables['rho']
    b = variables['b']
    rho_i = variables['rho_i']
    h = variables['h']
    H = variables['H']
    g = variables['g']
    L = variables['L']
    gaus = variables['gaus']
    # fname = 'Rectangle'
    # arg = [0.395,1.215]
    
    if arg == 'gaus':
        gaus = variables['gaus'] # its understod that if nothing else is mentioned then we use the standard gaussian distriution
        arg = gaus
        # print('gaus = ' + str(gaus))
    dx = l*2/m
    
    #-------------fourier grid--------------
    freq = fft.fftfreq(m,dx/2/np.pi)
    
    ksi1,ksi2 = np.meshgrid(freq,freq)
    
    
    ksi1[:,0] = 0.00001
    ksi2[0,:] = 0.00001
    
    #-----------fourier operators-------------
    
    rr = ksi1**2 + ksi2**2
    
    Fkk = 1 + rho_i*h/rho*(1+(h**2)*rr/12)*np.sqrt(rr)*np.tanh(H*np.sqrt(rr))  
    
    G_0k = np.sqrt(rr)*np.tanh(H*np.sqrt(rr))
    
    Rk = b*G_0k/(2*rho*Fkk)
    
    Uk = np.sqrt(g*(1+L**4*(rr**2))*G_0k/Fkk - Rk**2)
    
    #------------------X,Y grid---------------
    
    u = np.linspace(-l,l,m)
    
    X,Y = np.meshgrid(u,u)
    # In theory we could use the same procedure for the hankel transform
    # print("inital funciton used:" + fname + str(arg))
    if fname in initialFuncs:
        Fb = initialFuncs[fname]
    else:
        print(fname + " is not found in initial funcs" + "valid names are:" + str(initialFuncs.keys()))
    
   
    if fname == 'Dirac':
        Fhatk = Fb(X,Y,arg)
    else:
        Fhatk = fft.fftn(Fb(X,Y,arg))
        Fhatk = Fhatk

    #---------------this is just the procedure of extracting all of the oppreators------------------
    Liste = [ dx , X , Y , ksi1 , ksi2 , Fhatk , G_0k , Fkk , Rk , Uk]
    
    names = ['dx','X','Y','ksi1','ksi2','Fhatk','G_0k','Fkk','Rk','Uk']
    
    operators = dict()
    
    for (name,item) in zip(names,Liste):
        operators[name] = item
    
    return operators




def CalculateStrain(Nk,l,m):
    
    
    
    operators = GetOperators(l,m)
    
    ksi1 = operators["ksi1"]
    ksi2 = operators["ksi2"]
    
    
        
    dN = [fft.ifftn(-ksi1*ksi1*Nk),fft.ifftn(-ksi2*ksi1*Nk),fft.ifftn(-ksi2*ksi1*Nk),fft.ifftn(-ksi2*ksi2*Nk)]
    dN = np.real(dN)
    stres = np.zeros((m,m))
    for i in range(0,m):

        for j in range(0,m):
            # need a new way of calculating the eigenvalues
            a = dN[0][i,j] + dN[3][i,j]
            
            det = dN[0][i,j]*dN[3][i,j] - dN[1][i,j]*dN[2][i,j]
                
            eigenvalues = np.array([a/2 + np.sqrt(a**2 - 4*det)/2 , a/2 - np.sqrt(a**2 - 4*det)/2])

            stres[i,j] = np.max(np.abs(eigenvalues))
        
    return stres



def LVin(x_0,v,t,l,m):
    
    operators = GetOperators(l,m)
    # variables = GetVariables()
    
    Uk = operators['Uk']
    Rk = operators['Rk']
    G_0k = operators['G_0k']
    Fhatk = operators['Fhatk']
    ksi1 = operators['ksi1']
    ksi2 = operators['ksi2']
    Fkk = operators['Fkk']
    dx = operators['dx']
    
    j = np.complex(0,1)
    
    front = G_0k*j*Fhatk/2/Uk/Fkk
    
    den1 = Rk - j*Uk - j*(v[0]*ksi1 + v[1]*ksi2)
    

    den2 = Rk + j*Uk - j*(v[0]*ksi1 + v[1]*ksi2)
    
    
    #-----------------------------
    I_1 = np.exp(-j*((v[0]*t + x_0[0])*ksi1 + (v[1]*t + x_0[1])*ksi2)) - np.exp(-t*(Rk - j*Uk))

    
    I_2 = np.exp(-j*((v[0]*t + x_0[0])*ksi1 + (v[1]*t + x_0[1])*ksi2)) - np.exp(-t*(Rk + j*Uk))

    
    I_1 = front*I_1/den1
              
    I_2 = front*I_2/den2
    
    return I_1,I_2
    
    
    
def LVinEnd(x_end,v,t,l,m):
    
    """ parameters: x_end, v , t,l,m returns: I_1,I_2 the integrals afther t at the end point, to get the solution eta take the first fft.ifftn(I[0]-I[1])"""
    x_0 = -v*t+x_end
    
    return LVin(x_0,v,t,l,m)


def PropagateI(I_1,I_2,t,l,m):
    """ Used to propagate the solution, mainly for the merging of solutions"""
    operators = GetOperators(l,m)
    # variables = GetVariables()
    
    Uk = operators['Uk']
    Rk = operators['Rk']
    G_0k = operators['G_0k']
    Fhatk = operators['Fhatk']
    ksi1 = operators['ksi1']
    ksi2 = operators['ksi2']
    Fkk = operators['Fkk']
    dx = operators['dx']
    
    
    
    I_1 = I_1*np.exp(-t*(Rk-np.complex(0,1)*Uk))
    
    I_2 = I_2*np.exp(-t*(Rk+np.complex(0,1)*Uk))
    
    return I_1,I_2
    
def Merge(I1,T1,I2,T2,x1_end,x2_start,l,m):
    
    dif = x1_end-x2_start
    op = GetOperators(l,m)
    ksi1,ksi2 = op['ksi1'],op['ksi2']
    
    tl = np.exp(np.complex(0,1)*(dif[0]*ksi1 +dif[1]*ksi2))
    
    # print('tl=' + str(tl))
    # print(str(len(I1)))
    
    I1 =  [[k[0]*tl,k[1]*tl] for k in I1]
    
    
    
    I0 = I1[-1]
    
    I2 = GlueSol(I0,I2,T2,l,m)
    
    Iend = I1
    
    tend = T1[-1]
    
    T22 = [k+tend for k in T2]
    
    # here we just scip the first iniex and hope that it treats the whole thing as a list
    
    
    Tend = list(T1) + list(T22[1:])
    
    [Iend.append(k) for k in I2]
    
    return Tend,Iend
    
def GlueSol(Istart,Iend,T,l,m):
    """ Here Istart should be [I_1,I_2] and Iend should be a list of integrals where all coresponds to the T"""
    Ifin = []
    
    for t,II in zip(T[1:],Iend):
        prop = PropagateI(Istart[0],Istart[1],t,l,m)

        Ifin.append([II[0] + prop[0],II[1] + prop[1]])
        
    return Ifin


def GP(X_1,X_2,l,m,T):
    """ This is a general purpos function taking in the parameters X1 , X2 the functios describing the path taken by the load 
    l,m is the grid size, T is vector of timesteps"""
    operators = GetOperators(l,m)    
    variables = GetVariables()

    
    Uk = operators['Uk']
    Rk = operators['Rk']
    G_0k = operators['G_0k']
    Fhatk = operators['Fhatk']
    ksi1 = operators['ksi1']
    ksi2 = operators['ksi2']
    Fkk = operators['Fkk']

    
    front = G_0k*complex(0,1)*Fhatk/2/Uk/Fkk
    # This is the vector with the integrals
    I = []
    t_0 = T[0]
    sum1 = 0 
    sum2 = 0
    for t in T[1:]:
        
        ii1 = np.exp((t_0-t)*(Rk-complex(0,1)*Uk))
        ii2 = np.exp((t_0-t)*(Rk+complex(0,1)*Uk))

        print("t = " + str(t) + "t_end = " + str(T[-1]))
        
        @jit()
        def i_1(tau):
            return front*np.exp(-(t-tau)*(Rk-complex(0,1)*Uk) - complex(0,1)*(X_1(tau)*ksi1 + X_2(tau)*ksi2))
        @jit()
        def i_2(tau):
            return front*np.exp(-(t-tau)*(Rk+complex(0,1)*Uk) - complex(0,1)*(X_1(tau)*ksi1 + X_2(tau)*ksi2))
        
        print("start integrating ") 
        tt = time.time()
        I_1,err1,info1 = quad_vec(i_1,t_0,t,full_output=True,epsrel = 1e-14,epsabs = 0, workers = 1)
        print("I_1:" + str(time.time()-tt) + "s " + "err = " + str(err1))
        tt = time.time()
        I_2,err2,info2 = quad_vec(i_2,t_0,t,full_output=True,epsrel = 1e-14,epsabs = 0,workers = 1)
        print("I_2:" + str(time.time()-tt)+ "s " + "err = " + str(err2))
        
        sum1 = sum1*ii1 + I_1
        sum2 = sum2*ii2 + I_2
        
        I.append([sum1,sum2])
        t_0 = t
        
        
    return I

def EtaofI(I,n = 1):
    """returns the displacement eta, from the two integrals, using the methods returning a time series set n = 2"""
    if n == 1:
        return fft.ifftn(I[0]-I[1])
    if n == 2:
        return [fft.ifftn(k[0]-k[1]) for k in I]
    
    raise Exception("the number n:" + str(n) + "does not correspond to the datatype try n = 1,2. Or the format of I is not correct"  )
