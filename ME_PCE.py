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
    
    def __init__(self, PCE_method, d, p, B_init, fun, alg_mod, data_fun, N_t, N_p, theta1 = 0.00001, theta2 = 0.00001, alpha = 0.5, arg1 = 0.01, arg2 = 0.0001, arg3 = 0.01, arg4= 0.0001, sigma_vals = None, mu_vals = None, n_iter = 2):
        """
        PCE_method: 'aPCE' or 'aPCE_Stieltjes' or 'PCE_Legendre' or 'PCE_Hermite'
        alg_mod: AFVB_PCE or VRVM_PCE
        data_fun: function to generate data

        """
        
        self.data_fun = data_fun
        self.B_init = B_init
        self.fun = fun
        self.alg_mod = alg_mod
        self.PCE_method = PCE_method
        self.p = p
        self.d = d
        self.theta1 = theta1
        self.theta2 = theta2
        self.alpha = alpha
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.sigma_vals = sigma_vals
        self.mu_vals = mu_vals
        self.n_iter = n_iter
        self.N_p = N_p
        self.N_t = N_t
        
    def multivariate_pce_index(self, d, max_deg):
        """
        Generate all the d-dimensional polynomial indices with the 
        constraint that the sum of the indexes is <= max_deg

        input:
        d: int, number of random variables
        max_deg: int, the max degree allowed

        return: 
        2d array with shape[1] equal to d, the multivariate indices
        """
        maxRange = max_deg*np.ones(d, dtype = 'int')
        index = np.array([i for i in product(*(range(i + 1) for i in maxRange)) if sum(i) <= max_deg])

        return index
    
    
    def split_data(self, X, B_k):
        """
        X: data to split
        B_k: domains, needs to be an array of size d x 2
        """
        
        B = np.array(B_k)

        N = X.shape[0]
        d = X.shape[1]
        X_k = []

        for i in range(N):
            if np.sum((B[:,0] <= X[i]) & (X[i] <= B[:,1])) == self.d:
                X_k.append(list(X[i]))
                
        return np.array(X_k)
    
    def map_domain_to_negOne_One(self, B_k, domain_init):

        domain_init = np.array(domain_init)[0]
        a = np.c_[domain_init[:,0]]
        b = np.c_[domain_init[:,1]]

        B_k_new = 2/(b-a)*(B_k-(b+a)/2)

        return B_k_new
    
    def split_domain(self, N_t, N_p, theta1, theta2, alpha, iter_num):
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
        
        for j in range(iter_num):
            B_k = []
                
            for k in range(len(B)):   
                if (self.PCE_method == 'aPCE'):
                    X_p = self.data_fun(N_p, self.d, B[k])
                    mod = aPCE(X_p, self.p)
                    P = mod.Create_Orthonormal_Polynomials(self.p)  ######
                elif (self.PCE_method == 'aPCE_Stieltjes'):
                    X_p = self.data_fun(N_p, self.d, B[k])
                    mod = aPCE(X_p, self.p)
                    P = mod.Create_Orthonormal_Polynomials_Stieltjes(self.p)
                else:
                    mod = None
                    P = None
                    
                X_t = self.data_fun(N_t, self.d, B[k])
                Y_t = self.fun(X_t)

                model = self.alg_mod(self.PCE_method, self.d, self.p, B[k], mod, P, self.arg1, self.arg2, self.arg3, self.arg4, sigma_vals = self.sigma_vals, mu_vals = self.mu_vals).fit(X_t, Y_t.reshape(X_t.shape[0]))

                a_vec = model.a_full

                # For each random element k, calculate eta_k and J_k
                N_p_1 = int(math.factorial(self.d + self.p-1)/(math.factorial(self.d)*math.factorial(self.p-1)))
                eta_k = np.sum(a_vec[N_p_1:]**2)/np.sum(a_vec[1:]**2) 
                J_k = 1


                for i in range(self.d):

                    Bk_trans = self.map_domain_to_negOne_One(B[k], self.B_init)
                    J_k *= (Bk_trans[i][1] - Bk_trans[i][0])/2

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

                if (eta_k**alpha*J_k >= theta1): #& (X_t.shape[0]/2**np.sum(theta2*np.max(r)<=r) >= size_restriction):
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

            # stop the iteration if no splitting is to occur next
            if B == B_k:
                break

            B = B_k.copy()
            
        return np.array(B)

    def fit(self, X_train, Y_train):
        """
        X: Training Data
        """
         
        self.idx = self.multivariate_pce_index(self.d, self.p)
        self.n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))
        
        model_local = []
        P_local = []
        mod_local = []
        mean_local = []
        v_local = []
        a_local = []
        a_local_full = []
        active_cols_local = []
        Jk_i = []
        Jk = []
        n_star_local = []
        
        if (self.n_iter == 0):
            B = self.B_init
        else:
            B = self.split_domain(self.N_t, self.N_p, self.theta1, self.theta2, self.alpha, self.n_iter)

        for k in range(len(B)):
            
            if (self.PCE_method == 'aPCE'):    
                X_p = self.data_fun(self.N_p, self.d, B[k])
                mod = aPCE(X_p, self.p)
                mod_local.append(mod)
                P = mod.Create_Orthonormal_Polynomials(self.p)
                P_local.append(P)
                
            elif (self.PCE_method == 'aPCE_Stieltjes'):    
                X_p = self.data_fun(self.N_p, self.d, B[k])
                mod = aPCE(X_p, self.p)
                mod_local.append(mod)
                P = mod.Create_Orthonormal_Polynomials_Stieltjes(self.p)
                P_local.append(P)
            else:
                mod = None
                P = None
         
            #if (self.n_iter == 0):
                #X_t = self.split_data(X_train, B[k])
            #else:
            X_t = self.data_fun(self.N_t, self.d, B[k])
            #print(X_t.shape[0], 'k =', k)
            Y_t = self.fun(X_t)

            model = self.alg_mod(self.PCE_method, self.d, self.p, B[k], mod, P, self.arg1, self.arg2, self.arg3, self.arg4, sigma_vals = self.sigma_vals, mu_vals = self.mu_vals).fit(X_t, Y_t.reshape(X_t.shape[0]))
            
            J_k = 1
            Jk_i_temp = []
            for i in range(self.d):
                Bk_trans = self.map_domain_to_negOne_One(B[k], self.B_init)
                J_k *= (Bk_trans[i][1] - Bk_trans[i][0])/2
                Jk_i_temp.append((Bk_trans[i][1] - Bk_trans[i][0])/2)

            Jk_i.append(Jk_i_temp)

            Jk.append(J_k)
            model_local.append(model)
            if model.a_hat.shape[0] == 0:
                mean_local.append(float(model.a_full[0]))
            else:
                mean_local.append(float(model.a_hat[0]))
            v_local.append(np.sum(model.a_hat[1:]**2))
            a_local.append(model.a_hat)
            active_cols_local.append(model.active_cols)
            n_star_local.append(model.n_star)
            a_local_full.append(model.a_full)

        self.B_split = B
        self.P_local = P_local
        self.model_local = model_local
        self.mod_local = mod_local
        self.mean_local = np.array(mean_local)
        self.a_local = a_local
        self.v_local = np.array(v_local)
        self.active_cols_local = active_cols_local
        self.Jk = np.array(Jk)
        self.n_star_local = n_star_local
        self.a_local_full = a_local_full
        self.Jk_i = np.array(Jk_i)
        
        return self
        
        
    def predict(self, X_test, sparse = True):

        # Add condition that checks if test data is outside of the B_split domain
        Y = np.zeros(X_test.shape[0])
        for j in range(Y.shape[0]):
            for i in range(len(self.B_split)):

                if np.sum((self.B_split[i][:,0] <= X_test[j]) & (X_test[j] <= self.B_split[i][:,1])) == self.d:
                        
                    Y[j] += self.model_local[i].predict(X_test[j:j+1], sparse)

                    
        return Y
                    
                    
                    
                    