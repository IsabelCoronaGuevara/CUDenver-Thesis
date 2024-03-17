"""
Author: 
Date: 
"""

import numpy as np
import math
from itertools import product
from scipy.special import gamma
import sys
from sklearn.base import BaseEstimator
from scipy.special import legendre
from aPCE import *
from basis_functions import *

class AFVB_PCE(BaseEstimator):
    
    
    def __init__(self, PCE_method, d, p = 8, domain = None, aPCE_model = None, P = None, A_0 = 0.01, B_0 = 0.0001, C_0 = 0.01, D_0 = 0.0001, T_L = 0.001, eps = 1000, sigma_vals = None, mu_vals = None):
        """
        Initializes the object
        """
        
        self.p = p
        self.d = d
        self.A_0 = A_0
        self.B_0 = B_0
        self.C_0 = C_0
        self.D_0 = D_0
        
        self.T_L = T_L
        self.eps = eps
        self.aPCE_model = aPCE_model
        self.PCE_method = PCE_method
        self.P = P
        self.domain = domain
        self.sigma_vals = sigma_vals
        self.mu_vals = mu_vals
        
        if (PCE_method == 'aPCE'):
            self.basis = basis(self.d, self.p, self.domain, self.aPCE_model, self.P).basis_aPCE
            
        elif (PCE_method == 'PCE_Legendre'):
            self.basis = basis(self.d, self.p, self.domain).basis_PCE_Legendre
        
        elif (PCE_method == 'PCE_Hermite'):
            self.basis = basis(self.d, self.p, self.domain, self.aPCE_model, self.P, self.sigma_vals, self.mu_vals).basis_PCE_Hermite
        
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
        
    def fit(self, X, Y):
        
        Phi_in = self.basis(X)
        self.N = Phi_in.shape[0]
        self.n = Phi_in.shape[1]
     
            
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
        col_full = []
        col = np.array(range(Phi_in.shape[1]))
        col_full.append(col)
        a_all = []
        chi_all = []
        Br_all = []
        
        Beta = 1
        L_old = None

        Phi = Phi_in
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
                    chi_r_inv_times_a_r += np.c_[Phi[i,:]]*Y[i]

                a_r = chi_r@chi_r_inv_times_a_r

                #########################################
                #### Update B_r, D_r, E_Theta, ell_r ####
                #########################################

                #### B_r
                B_r = self.B_0 + 1/2*(np.sum(Y**2) - a_r.transpose()@chi_r_inv@a_r)

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
                    s_2 += A_r/B_r*(Y[i] - np.c_[Phi[i,:]].transpose()@a_r)**2 \
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
            
            #print(Beta, Phi.shape, float(L_r))
            ########################################################################################
            ############################### Save the optimal vectors ###############################
            ########################################################################################
            if Beta == 1:
                Phi_full = Phi_in
                a_full = a_r
                a_all.append(a_full)
                chi_all.append(chi_r)
                Br_all.append(B_r)
                

            #if k == 1:    
            #    print('Beta = ',Beta, '', 'n = ', Phi.shape[1], 'VLB = ', float(L_r))
            ########################################################################################
            ########################################################################################

            #### Compute lambda_beta ####
            lambda_beta = np.diag(np.linalg.inv(E_Theta))

            #### Compute ln_T_beta ####
            ln_T_beta = min(np.log(lambda_beta)) + (max(np.log(lambda_beta)) - min(np.log(lambda_beta)))/self.eps

            ########################################################################################
            ############################# Update columns we are keeping ############################
            ########################################################################################
            col = col[np.log(lambda_beta) >= ln_T_beta]
            col_full.append(col)
            
            a_all.append(a_r)
            chi_all.append(chi_r)
            Br_all.append(B_r)
         
            ########################################################################################
            ########################################################################################

            #### Prune Phi and E_Theta
            Phi = Phi[:,np.log(lambda_beta) >= ln_T_beta]
            E_Theta = E_Theta[:, np.log(lambda_beta) >= ln_T_beta]
            E_Theta = E_Theta[np.log(lambda_beta) >= ln_T_beta,:]

            Beta += 1
        
        max_beta = np.argmax(L_beta) + 1
        
        self.active_cols = col_full[max_beta-1]    
        self.Phi_hat = Phi_full[:, self.active_cols]
        self.a_hat = a_all[max_beta]
        self.Phi_full = Phi_full
        self.a_full = a_full
        self.n_star = self.active_cols.shape[0]
        self.chi = chi_all[max_beta]
        self.Br = Br_all[max_beta]
        self.chi_full = chi_all[0]
        
        #print('Beta_star = ', max_beta,'', 'n_star = ', self.n_star)
                
        return self
    
    def predict(self, X, sparse = True):
        if sparse is True:
            return self.basis(X)[:,self.active_cols]@self.a_hat
        elif sparse is False:
            return self.basis(X)@self.a_full


