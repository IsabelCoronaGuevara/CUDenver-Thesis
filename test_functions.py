"""
The module contains definitions for test functions

__author__ = 'Isabel Corona Guevara'

"""

import numpy as np
import math

    
def ishigami(X, a = 7, b = 0.1):
    """
    Evaluate the ishigami function. 

    Inputs:
    X: n-dim array of size (n_sample, 3), in the range of [-pi, pi]
    a: double, value a
    b: double, value b

    Returns:
    1-dim array, the values of the ishigami function
    """

    val = np.c_[np.sin(X[:,0]) + a*(np.sin(X[:,1]))**2 + b*(X[:,2])**4*np.sin(X[:,0])]

    return val 


def morris(X):
    """
    Evaluate the morris function. 

    Inputs:
    X: n-dim array of size (n_sample, 20), in the range of [0, 1]

    Returns:
    1-dim array, the values of the morris function
    """

    N = X.shape[0]
    d = X.shape[1]

    # Making indexes for the sums
    index2 = []
    for j in range(1,20):
        for i in range(j):
            index2.append([i,j])
    index2 = np.array(index2)

    index3 = []
    for l in range(2,5):
        for j in range(1,l):
            for i in range(j):
                index3.append([i,j,l])
    index3 = np.array(index3)

    # Creating omega
    omega = np.zeros((N, d))
    for i in range(d):
        omega[:,i] = 2*X[:,i] - 1
    for i in [2,4,6]:
        omega[:,i] = 2.4*X[:,i]/(X[:,i] + 1) - 1

    # Sum with one index
    s1_1_to_10 = np.sum(20*omega[:,0:10], axis = 1)
    s1_11_to_20 = np.sum(omega[:,10:20], axis = 1)

    s1 = s1_1_to_10 + s1_11_to_20

    # Sum with two index
    s2_1_to_6 = np.zeros(N)
    s2_7_to_20 = np.zeros(N)
    for k in range(index2.shape[0]):
        if index2[k][1] <= 5:
            s2_1_to_6 += -15*omega[:,index2[k][0]]*omega[:,index2[k][1]]
        else:
            s2_7_to_20 += (-1)**(index2[k][0] + index2[k][1] + 2)\
            *omega[:,index2[k][0]]*omega[:,index2[k][1]]

    s2 = s2_1_to_6 + s2_7_to_20

    # Sum with three index
    s3 = np.zeros(N)
    for k in range(index3.shape[0]):
        s3 += (-10)*omega[:,index3[k][0]]*omega[:,index3[k][1]]*omega[:,index3[k][2]]

    # Sum with four index
    s4 = 5*omega[:,0]*omega[:,1]*omega[:,2]*omega[:,3]

    return s1 + s2 + s3 + s4


def borehole(X):
    """
    Evaluate the borehole function. 

    Inputs:
    X: n-dim array of size (n_sample, 8)
        * x1 ~ N(0.1, 0.0161812)
        * x2 ~ Lognormal(7.71, 1.0056)
        * x3 ~ U(63070, 115600)
        * x4 ~ U(990, 1110)
        * x5 ~ U(63.1, 116)
        * x6 ~ U(700, 820)
        * x7 ~ U(1120, 1680)
        * x8 ~ U(9855, 12045)

    Returns:
    1-dim array, the values of the borehole function
    """

    r_w = X[:,0]
    r = X[:,1]
    T_u = X[:,2]
    H_u = X[:,3]
    T_l = X[:,4]
    H_l = X[:,5]
    L = X[:,6]
    K_w = X[:,7]

    val = (2*np.pi*T_u*(H_u-H_l))/(np.log(r/r_w)*(1 + (2*L*T_u)/(np.log(r/r_w)*r_w**2*K_w) + T_u/T_l))

    return val


def borehole_data(N):
    """
    Simulate data to be used for the borehole function

    Inputs:
    N: sample data size

    Returns:
    n-dim array, data of size (n_samples, 8)
    """

    X = np.zeros((N, 8))

    X[:,0] = np.random.normal(0.1, 0.0161812, size = N)
    X[:,1] = np.random.lognormal(7.71, 1.0056, size = N)
    X[:,2] = np.random.uniform(63070, 115600, size = N)
    X[:,3] = np.random.uniform(990, 1110, size = N)
    X[:,4] = np.random.uniform(63.1, 116, size = N)
    X[:,5] = np.random.uniform(700, 820, size = N)
    X[:,6] = np.random.uniform(1120, 1680, size = N)
    X[:,7] = np.random.uniform(9855, 12045, size = N)

    return X


def one_dof_oscillator(X):
    """
    Evaluate the 1-dof undamped oscillator function. 

    Inputs:
    X: n-dim array of size (n_sample, 6)
        * x1 ~ LogNormal(1, 0.15^2) (Mass) 
        * x2 ~ LogNormal(1, 0.2^2) && (Stiffness of spring one)
        * x3 ~ LogNormal(0.1, 0.02^2) && (Stiffness of spring two)
        * x4 ~ LogNormal(0.5, 0.1^2) && (Displacement yielded by one of the springs)
        * x5 ~ LogNormal(1, 0.2^2) && (Exciting force)
        * x6 ~ LogNormal(1, 0.1^2) && (Excitation time)

    Returns:
    1-dim array, the values of the 1-dof undamped oscillator function
    """

    m = X[:,0]
    c1 = X[:,1]
    c2 = X[:,2]
    r = X[:,3]
    F1 = X[:,4]
    t1 = X[:,5]

    w0 = np.sqrt((c1+c2)/m)

    val = 3*r - np.abs(2*F1/(m*w0**2)*np.sin(w0**2*t1/2))

    return val 


def one_dof_oscillator_data(N):
    """
    Simulate data to be used for the 1-dof undamped oscillator function

    Inputs:
    N: sample data size

    Returns:
    n-dim array, data of size (n_samples, 6)
    """

    X = np.zeros((N, 6))

    X[:,0] = np.random.lognormal(1, 0.15, size = N)
    X[:,1] = np.random.lognormal(1, 0.2, size = N)
    X[:,2] = np.random.lognormal(0.1, 0.02, size = N)
    X[:,3] = np.random.lognormal(0.5, 0.1, size = N)
    X[:,4] = np.random.lognormal(1, 0.2, size = N)
    X[:,5] = np.random.lognormal(1, 0.1, size = N)

    return X


def cantilever(X):
    """
    Evaluate the cantilever function. 

    Inputs:
    X: n-dim array of size (n_sample, 6)
        * x1 ~ N(42, 0.4998^2) (mm)
        * x2 ~ N(5, 0.1^2) (mm)
        * x3 ~ N(560, 56^2) (MPa)
        * x4 ~ N(1800, 180^2) (N)
        * x5 ~ N(1800, 180^2) (N)
        * x6 ~ N(1000, 100^2) (N)
        * x7 ~ N(1900, 190^2) (Nm) 

    Returns:
    1-dim array, the values of the cantilever function
    """

    D = X[:,0]
    h = X[:,1]
    R0 = X[:,2]
    F1 = X[:,3]
    F2 = X[:,4]
    P = X[:,5]
    T = X[:,6]

    L1 = 60
    L2 = 120
    Theta1 = np.pi/18
    Theta2 = np.pi/36


    M = F1*np.cos(Theta1)*L1 + F2*np.cos(Theta2)*L2
    A = np.pi/4*(D**2 - (D - 2*h)**2)
    I = np.pi/64*(D**4 - (D - 2*h)**4)
    sigma_x = (F1*np.sin(Theta1) + F2*np.sin(Theta2) + P)/A + M*D/(2*I)
    tau_zx = T*D/(4*I)

    val = R0/np.sqrt(sigma_x**2 + 3*tau_zx**2) # Units in MPa = N/mm^2

    return val


def cantilever_data(N):
    """
    Simulate data to be used for the cantilever function

    Inputs:
    N: sample data size

    Returns:
    n-dim array, data of size (n_samples, 7)
    """

    X = np.zeros((N, 7))

    D_mu = 42
    h_mu = 5
    R0_mu = 560
    F_mu = 1800
    P_mu = 1000
    T_mu = 1900

    # D, h, R0, F1, F2, P, T
    X[:,0] = np.random.normal(D_mu, D_mu*0.0119, N)
    X[:,1] = np.random.normal(h_mu, h_mu*0.02, N)
    X[:,2] = np.random.normal(R0_mu, R0_mu*0.1, N)
    X[:,3] = np.random.normal(F_mu, F_mu*0.1, N)
    X[:,4] = np.random.normal(F_mu, F_mu*0.1, N)
    X[:,5] = np.random.normal(P_mu, P_mu*0.1, N)
    X[:,6] = np.random.normal(T_mu, T_mu*0.1, N)

    return X


def sobol(X, d, a):
    """
    Evaluate the sobol function. 

    Inputs:
    X: n-dim array of size (n_sample, d), in the range of [0, 1]
    d: integer, number of random variables
    a: 1-dim array of size d, coefficients

    Returns:
    1-dim array, the values of the sobol function
    """

    val = np.c_[np.prod((np.abs(4*X - 2) + alpha)/(1 + alpha),1)]

    return val




