"""
The module contains methods for GSA including approximations using MC and PCE-based GSA.

__author__ = 'Isabel Corona Guevara'

"""

import numpy as np
import math
import pandas as pd
from itertools import product

    
class sobol_GSA(object):
    """
    Calculates the Sobol Indices for GSA using MC
    """
    
    def __init__(self, d, fun):
        
        self.d = d
        self.fun = fun
        
        
        
    def create_dataAB(self, dataA, dataB, variable_index_to_fix):
        """
        """
        
        dataB_withA = dataB.copy()
        dataB_withA[:, variable_index_to_fix] = dataA[:, variable_index_to_fix]
        
        return dataB_withA

    
    def sobol_indice_1st_and_total_order(self, variable_index, dataA, dataB):
        """
        """

        dataB_withA = self.create_dataAB(dataA, dataB, variable_index)

        N = len(dataA)

        y_A = self.fun(dataA)
        y_AB = self.fun(dataB_withA)
        y_B = self.fun(dataB)
        V = np.var(self.fun(dataA))

        num_1st_order = N*np.sum(np.multiply(y_A,y_AB)) - (np.sum(y_A)*np.sum(y_AB))
        num_tot = N*np.sum(np.multiply(y_B,y_AB)) - (np.sum(y_A)**2)
        deno = N*np.sum(y_A**2) - (np.sum(y_A))**2

        return np.round(V*num_1st_order/deno, 8), np.round(V*(1 - (num_tot/deno)), 8), np.round(num_1st_order/deno, 8), np.round((1 - (num_tot/deno)), 8), np.round(V, 8)

    
    def sobol_MC(self, dataA, dataB):
        """
        
        MC_data: generated data for MC, ex., MC_data = np.random.uniform(-np.pi, np.pi, size=(N_mc, self.d))
        """
        
        V_partial = []
        V_total = []
        st_order = []
        total = []
        
        for i in range(self.d):
            results = self.sobol_indice_1st_and_total_order(i, dataA, dataB)
            V_partial.append(results[0])
            V_total.append(results[1])
            st_order.append(results[2])
            total.append(results[3])
            V = results[4]

        df_result = pd.DataFrame({
            "Partial Variance" : V_partial,
            "Total Variance" : V_total,
            "1st Order" : st_order,
            "Total Order" : total
            })

        return df_result, V
    
    
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
    

    def sobol_PCE(self, p, a, active_cols):
        """
        """
        
        
        V = np.sum(a[1:]**2)
        V_Partial = []
        V_Total = []
        S_Partial = []
        S_Total = []
        
        idx = self.multivariate_pce_index(self.d, p)

        for k in range(self.d):
            l = list(range(self.d))
            l.remove(k)

            temp = np.full(idx[active_cols].shape[0], True, dtype = bool)

            for i in range(idx[active_cols].shape[0]):
                temp[i] = (idx[active_cols][i][k] != 0) & (sum(idx[active_cols][i][l]) == 0)

            V_Partial.append(np.sum(a[temp]**2))
            S_Partial.append(np.sum(a[temp]**2)/V)

            for i in range(idx[active_cols].shape[0]):
                temp[i] = idx[active_cols][i][k] != 0

            V_Total.append(np.sum(a[temp]**2))
            S_Total.append(np.sum(a[temp]**2)/V)

            GSA_df = pd.DataFrame({
                #"variable" : list(range(1,d+1)),
                "Partial Variance" : V_Partial,
                "Total Variance" : V_Total,
                "1st Order" : S_Partial,
                "Total Order" : S_Total
            })
            
        return GSA_df, V
            
    def sobol_ME(self, p, a, active_cols, global_mean, global_V):
        """
        """
        
        V = global_V #np.sum(a[1:]**2)
        V_Partial = []
        V_Total = []
        S_Partial = []
        S_Total = []
        
        idx = self.multivariate_pce_index(self.d, p)

        for k in range(self.d):
            l = list(range(self.d))
            l.remove(k)

            temp = np.full(idx[active_cols].shape[0], True, dtype = bool)

            for i in range(idx[active_cols].shape[0]):
                temp[i] = (idx[active_cols][i][k] != 0) & (sum(idx[active_cols][i][l]) == 0)

            V_Partial.append(np.sum(a[temp]**2))
            S_Partial.append(np.sum(a[temp]**2)/V)

            for i in range(idx[active_cols].shape[0]):
                temp[i] = idx[active_cols][i][k] != 0
                
            V_Total.append(np.sum(a[temp]**2))
            S_Total.append(np.sum(a[temp]**2)/V)

            GSA_df = pd.DataFrame({
                #"variable" : list(range(1,d+1)),
                "Partial Variance" : V_Partial,
                "Total Variance" : V_Total,
                "1st Order" : S_Partial,
                "Total Order" : S_Total
            })


        return GSA_df, V
        
        
    
    