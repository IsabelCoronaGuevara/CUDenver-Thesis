#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Author: 
Date: 
"""

__all__ = ['AFVB_PCE']

import numpy as np
import math
from itertools import product
from scipy.special import gamma
import sys

class AFVB_PCE(object):
    
    
    def __init__(self, X, Y, Phi, p, A_0, B_0, C_0, D_0, T_L, eps):
        """
        Initializes the object
        """
        self.N = X.shape[0]
        self.n = int(math.factorial(X.shape[1] + p)/(math.factorial(X.shape[1])*math.factorial(p)))
        
        self.Y = Y
        self.Phi = Phi
        
        self.A_0 = A_0
        self.B_0 = B_0
        self.C_0 = C_0
        self.D_0 = D_0
        
        self.T_L = T_L
        self.eps = eps
        
    def algorithm(self):
        
        for k in range(2): # Running loop twice, second time will store optimal values
            if k == 1:
                max_beta = np.argmax(L_beta) + 1
                print('Beta_star = ', max_beta)
            else:
                max_beta = 0
            
            ########################################
            #### Compute initial E_theta[Theta] ####
            ########################################
            E_Theta = np.zeros((self.n, self.n))
            
            for i in range(self.n):
                E_Theta[i,i] = self.C_0/self.D_0

            #### Compute A_r and C_r
            A_r = self.A_0 + self.N/2
            C_r = self.C_0 + 1/2

            #### List that keeps track of L_beta values and corresponding Beta value
            L_beta = []

            ### An array made in order to keep track of which column numbers
            ### Phi_hat kept in the end
            col = np.array(range(0,self.Phi.shape[1]))

            Beta = 1
            L_old = None
            
            Phi = self.Phi
            while Phi.shape[1] > 1: ## Beta 
                n = Phi.shape[1]

                e_L = sys.float_info.max
                r = 1

                while e_L > self.T_L:
                    ###############################
                    #### Compute chi_r and a_r ####
                    ###############################

                    #### chi_r
                    chi_r_inv = np.zeros((n, n))

                    for i in range(self.N):
                        chi_r_inv += np.c_[Phi[i,:]]@np.c_[Phi[i,:]].transpose()

                    chi_r_inv += E_Theta
                    chi_r = np.linalg.inv(chi_r_inv)


                    #### a_r
                    chi_r_inv_times_a_r = np.zeros((n, 1))
                    for i in range(self.N):
                        chi_r_inv_times_a_r += np.c_[Phi[i,:]]*self.Y[i]

                    a_r = chi_r@chi_r_inv_times_a_r

                    #########################################
                    #### Update B_r, D_r, E_Theta, ell_r ####
                    #########################################

                    #### B_r
                    B_r = self.B_0 + 1/2*(np.sum(self.Y**2) - a_r.transpose()@chi_r_inv@a_r)

                    #### D_r
                    D_r = np.zeros((n,1))
                    for i in range(n):
                        D_r[i] = self.D_0 + 1/2*(a_r[i]**2*A_r/B_r + chi_r[i,i])         

                    #### E_Theta
                    for i in range(n):
                        E_Theta[i,i] = float(C_r/D_r[i])

                    #### L_r
                    s_2 = 0
                    for i in range(self.N):
                        s_2 += A_r/B_r*(self.Y[i] - np.c_[Phi[i,:]].transpose()@a_r)**2 \
                        + np.c_[Phi[i,:]].transpose()@chi_r@np.c_[Phi[i,:]]

                    det_chi_r = np.linalg.det(chi_r)  # determinant of Xr
                    if det_chi_r == 0.0:
                        det_chi_r = np.finfo(np.float64).tiny  
                           
                    L_r = -self.N/2*math.log(2*math.pi) \
                        - 1/2*s_2 \
                        + math.log(gamma(A_r)) \
                        + A_r*(1 - math.log(B_r) - self.B_0/B_r) \
                        - math.log(gamma(self.A_0)) \
                        + self.A_0*math.log(self.B_0) \
                        - np.sum(C_r*np.log(D_r)) \
                        + n*(1/2 - math.log(gamma(self.C_0)) + self.C_0*math.log(self.D_0) + math.log(gamma(C_r))) \
                        + 1/2*math.log(det_chi_r)

                    
                    ##############################
                    ####    Compute e_ell     ####
                    ##############################
                    if L_old is None:
                        L_old = L_r
                    else:
                        e_L = np.abs(100*(float(L_r) - L_old)/L_old)
                        L_old = float(L_r)
                        
                    r += 1   

                #### Save L_r value in an array
                L_beta.append(float(L_r))

                ########################################################################################
                ############################### Save the optimal vectors ###############################
                ########################################################################################
                if Beta == 1:
                    Phi_full = self.Phi
                    a_full = a_r
                if Beta == max_beta:
                    Phi_hat = Phi
                    a_hat = a_r
                if k == 1:    
                    print('Beta = ',Beta, '', 'n = ', Phi.shape[1], 'VLB = ', float(L_r))
                ########################################################################################
                ########################################################################################

                #### Compute lambda_beta ####
                lambda_beta = np.diag(np.linalg.inv(E_Theta))

                #### Compute ln_T_beta ####
                ln_T_beta = min(np.log(lambda_beta)) + (max(np.log(lambda_beta)) - min(np.log(lambda_beta)))/self.eps

                ########################################################################################
                ############################# Update columns we are keeping ############################
                ########################################################################################
                if Beta < max_beta:
                    col = col[np.log(lambda_beta) >= ln_T_beta]
                ########################################################################################
                ########################################################################################

                #### Prune Phi and E_Theta
                Phi = Phi[:,np.log(lambda_beta) >= ln_T_beta]
                E_Theta = E_Theta[:, np.log(lambda_beta) >= ln_T_beta]
                E_Theta = E_Theta[np.log(lambda_beta) >= ln_T_beta,:]

                Beta += 1

        return Phi_hat, a_hat, Phi_full, a_full, col
    

    
#class VRVM_PCE(object):
    
    
#    def __init__(self, X, Y, Phi, p, A_0, B_0, C_0, D_0, T_L, eps):
#        """
#        Initializes the object
#        """
#        self.N = X.shape[0]
#        self.n = int(math.factorial(X.shape[1] + p)/(math.factorial(X.shape[1])*math.factorial(p)))
#        
#        self.Y = Y
#        self.Phi = Phi
#        
#        self.A_0 = A_0
#        self.B_0 = B_0
#        self.C_0 = C_0
#        self.D_0 = D_0
#        
#        self.T_L = T_L
#        self.eps = eps
#        
#    def algorithm(self):


