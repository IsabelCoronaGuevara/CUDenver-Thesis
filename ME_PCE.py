"""
The module contains ...
Add description

__author__ = 'Isabel Corona Guevara'
__version__ = '0.1'
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import math
import scipy.integrate as integrate
from itertools import product
from sklearn.base import BaseEstimator
from aPCE import *
from AFVB_PCE import *
from VRVM_PCE import *

class ME_PCE(BaseEstimator):
    """
    Description Here
    
    """
    
    def __init__(self, X_pol, B_init, fun, idx, alg_mod, basis, PCE_method, p = 5, sparse = True, theta1 = 0.001, theta2 = 0.01, alpha = 1/2, size_restriction = 25, B_split = None, arg1 = 0.00001, arg2 = 0.00001, arg3 = 0.00001, arg4= 0.00001):
        """


        """
        
        self.X_pol = X_pol
        self.B_init = B_init
        self.fun = fun
        self.idx = idx
        self.alg_mod = alg_mod
        self.basis = basis
        self.PCE_method = PCE_method
        self.p = p
        self.sparse = sparse
        self.theta1 = theta1
        self.theta2 = theta2
        self.alpha = alpha
        self.size_restriction = size_restriction
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.B_split = B_split
    
    def split_data(self, X, B_k):
        """
        X: data to split
        B_k: domains, needs to be an array of size d x 2
        """
        
        B_k = np.array(B_k)

        N = X.shape[0]
        d = X.shape[1]
        X_k = []

        for i in range(N):
            if np.sum((B_k[:,0] <= X[i]) & (X[i] <= B_k[:,1])) == 3:
                X_k.append(list(X[i]))
                
        return np.c_[np.array(X_k)]
    
    def split_domain(self, X_train, X_pol, theta1, theta2, alpha, size_restriction = 25):
        """
        Inputs:
        
        theta1: 
        theta2:
        alpha: 0 < alpha < 1
        B_initial: Full Domain
        
        Return: 
        The B_ks
    
        Notes:
        Need to change a_full to a_hat later
        Need to be able to input P somehow
        """
        B = self.B_init
        
        for j in range(self.d):
            B_k = []
            for k in range(len(B)):

                X_p = self.split_data(X_pol, np.array(B[k]))
                X_t = self.split_data(X_train, np.array(B[k]))
                Y_t = self.fun(X_t)

                mod = aPCE(X_p, self.p, self.idx)
                P = mod.Create_Orthonormal_Polynomials(self.p)  ######

                model = self.alg_mod(self.basis, np.array(B[k]), self.PCE_method, mod, P, self.p, self.arg1, self.arg2, self.arg3, self.arg4).fit(X_t, Y_t.reshape(X_t.shape[0]))
                
                if self.sparse is True:
                    a_vec = np.zeros((self.n,1))
                    a_vec[model.active_cols] = np.c_[model.a_full[model.active_cols]]
                    
                elif self.sparse is False:
                    a_vec = model.a_full

                # For each random element k, calculate eta_k and J_k
                N_p_1 = int(math.factorial(self.d + self.p-1)/(math.factorial(self.d)*math.factorial(self.p-1)))
                eta_k = np.sum(a_vec[N_p_1:]**2)/np.sum(a_vec[1:]**2) 
                J_k = 1

                for i in range(self.d):
                    J_k *= (B[k][i][1] - B[k][i][0])/(2*np.pi)

                r = []
                for i in range(self.d):
                    l = [0]*self.d
                    l[i] = self.p
                    if (np.sum(a_vec[N_p_1:]**2)==0):
                        print('Cannot calculate r_i, setting it equal to 1')
                        r_i = 1
                    else:
                        r_i = float(a_vec[np.nonzero(np.sum(self.idx == l, 1) == self.d)])**2/np.sum(a_vec[N_p_1:]**2)
                    
                    r.append(r_i)

                if (eta_k**alpha*J_k >= theta1) & (X_t.shape[0]/2**np.sum(theta2*np.max(r)<=r) >= size_restriction):
                    # If this isn't true then we don't split up B_k
                    #print('Splitting', 'time =', j)

                    B_temp = []
                    #print(theta2*np.max(r)<=r, 'k =', k)

                    for i in range(self.d):
                        if (theta2*np.max(r)<=r[i]):
                            B_temp.append([[B[k][i][0], B[k][i][0] + (B[k][i][1]-B[k][i][0])/2], [B[k][i][0] + (B[k][i][1]-B[k][i][0])/2, B[k][i][1]]])
                        else:
                            B_temp.append([B[k][i], B[k][i]])

                    for i in range(self.d):
                        if (theta2*np.max(r)<=r[i]):
                            T_p = i
                            break

                    B_new = []

                    T_position = 0
                    # Fill B_new with the first element
                    if (theta2*np.max(r)<=r[0]):
                        for m in range(int(2**(i+1)/2**T_p)):
                            for l in range(int(2**np.sum((theta2*np.max(r)<=r))/(2**(i+1)/2**T_p))):
                                B_new.append([B_temp[0][m%2]])
                    else:
                        for l in range(int(2**np.sum((theta2*np.max(r)<=r)))):
                            B_new.append([B_temp[0][0]])

                    # Fill in for the rest of the elements
                    for i in range(1,self.d):
                        t = 0
                        if (theta2*np.max(r)<=r[i]):
                            for m in range(int(len(B_new)/2**T_position)):
                                for l in range(int(2**T_position)):
                                    B_new[t].append(B_temp[i][m%2])
                                    t += 1
                            T_position += 1
                        else:
                            for l in range(int(2**np.sum((theta2*np.max(r)<=r)))):
                                B_new[t].append(B_temp[i][0])
                                t += 1
                    #print(B_new)
                    B_k = B_k+B_new

                else:
                    #print('Not Splitting', 'time =', j, 'k =', k)
                    B_new = B[k]
                    #print(B_k)
                    B_k = B_k+[B_new]
                    #print(B_new)

            B = B_k.copy()
            
        return np.array(B)

    def fit(self, X_train, Y_train):
        """
        X: Training Data
        """
         
        self.N = X_train.shape[0] # sample size
        self.d = X_train.shape[1] # number of parameters
        self.n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))
        
        if self.B_split is None: # use this when we are optimizing theta1, theta2, alpha
            model_local = []
            P_local = []
            mod_local = []

            B = self.split_domain(X_train, self.X_pol, self.theta1, self.theta2, self.alpha, self.size_restriction)

            for k in range(len(B)):
                X_p = self.split_data(self.X_pol, B[k])
                X_t = self.split_data(X_train, B[k])
                #print(X_t.shape[0], 'k =', k)
                Y_t = self.fun(X_t)

                mod = aPCE(X_p, self.p, self.idx)
                mod_local.append(mod)
                P = mod.Create_Orthonormal_Polynomials(self.p)
                P_local.append(P)

                model = self.alg_mod(self.basis, B[k], self.PCE_method, mod, P, self.p, self.arg1, self.arg2, self.arg3, self.arg4).fit(X_t, Y_t.reshape(X_t.shape[0]))
                model_local.append(model)

            self.B_split = B
            self.P_local = P_local
            self.model_local = model_local
            self.mod_local = mod_local
        else: # use this when we are optimizing parameters of AFVB or VRVM. We already have arguments for preditions in this case.
            model_local = []
            P_local = []
            mod_local = []
            B = self.B_split
            
            for k in range(len(B)):
                X_p = self.split_data(self.X_pol, B[k])
                X_t = self.split_data(X_train, B[k])
                #print(X_t.shape[0], 'k =', k)
                Y_t = self.fun(X_t)

                mod = aPCE(X_p, self.p, self.idx)
                mod_local.append(mod)
                P = mod.Create_Orthonormal_Polynomials(self.p)
                P_local.append(P)

                model = self.alg_mod(self.basis, B[k], self.PCE_method, mod, P, self.p, self.arg1, self.arg2, self.arg3, self.arg4).fit(X_t, Y_t.reshape(X_t.shape[0]))
                model_local.append(model)

            self.P_local = P_local
            self.model_local = model_local
            self.mod_local = mod_local
           
        
    def predict(self, X_test, sparse = True):

        Y = np.zeros(X_test.shape[0])
        for j in range(Y.shape[0]):
            for i in range(len(self.B_split)):
                if np.sum((self.B_split[i][:,0] <= X_test[j]) & (X_test[j] <= self.B_split[i][:,1])) == self.d:
                    #mod = self.mod_local[i]
                    P = self.P_local[i]
                    Y[j] += self.model_local[i].predict(X_test[j:j+1], sparse)
                    
        return Y
                    
                    
                    
                    