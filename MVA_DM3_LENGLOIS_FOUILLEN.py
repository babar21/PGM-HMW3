#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:59:23 2018

@author: bastien
"""

#%%
# import modules
import pandas as pd
import os
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import seaborn as sns
import functools
import pickle

#%%
# data pre-processing

def pre_process(address):
    list_file = os.listdir(address)
#    list_file.remove('.DS_Store')
    dic = dict()
    for file in list_file:
        [a,b] = file.split('.')
        file_address = address + '/' + file
        df = pd.read_csv(file_address, delim_whitespace=True, header=None)
        l = len(df.columns)
        list_col = ['x_{}'.format(i+1) for i in range(l)]
        df.columns = list_col
        dic[b] =  df
    return dic


def import_init_data():
    with open('mu_init','rb') as fp:
        mu = pickle.load(fp)
    with open('sigma_sq_init','rb') as fp:
        sigma_sq = pickle.load(fp)
    return mu, sigma_sq
    

#%% Sum-Product algorithm

def normal_density(data, mu, sigma_sq):
    # data : i=1, j=nb_coordinates, k=nb_states(=1)
    # mu : i=1, j=nb_coordinates, k=nb_states
    # sigma_sq : i= , j= , k=nb_states 
    return  1./((2*np.pi)**(data.shape[1]/2.)*np.sqrt(np.linalg.det(sigma_sq.T)))*np.exp(-0.5*np.einsum('ilk,ilk->ik',
                      np.einsum('ilk,ljk->ijk',(data-mu),np.linalg.inv(sigma_sq.T).T),(data-mu)))


@functools.lru_cache()
def alpha_rec(t, data, mu, sigma_sq, A, pi_0, K):
    # defined in a recursive form, with vectorization (calculation for all states at once)
    # returns a vector with value log(alpha) for each state q_t
    # pi : 
    if t==0 :
        return np.einsum('j,ij->ij',pi_0,normal_density(np.expand_dims(data[t],axis=0), mu, sigma_sq))
    else :
        # use the sum-log-exp trick to avoid overflows
        s = np.einsum('ij,li->jl',np.exp(np.log(A)),np.exp(alpha_rec(t-1, data, mu, sigma_sq, A, pi_0, K)))
        
        return np.einsum('ij,ji->ji',s, normal_density(np.expand_dims(data[t],axis=0), mu, sigma_sq))


#@functools.lru_cache()
#def alpha_rec_bis(t, q_t, data, mu, sigma_sq, A, pi, K):
#    # defined in a recursive form, with vectorization
#    if t==0 :
#        return normal_density(np.expand_dims(data[t],axis=0), mu, pi, sigma_sq)
#    else :
#        s = 0
#        for i in range(K):
#            s += A[i,q_t]*alpha_rec(t-1, i, u, mean, covar, A, K)
#        return sps.multivariate_normal.pdf(u[0], mean=mean[q_t], cov=covar[q_t])*s
#

@functools.lru_cache()
def beta_rec(t, u, mean, covar, A, K, T):
    # defined in a recursive form, with vectorization (calculation for all states at once)
    # returns a vector with value log(beta) for each state q_t
    if t == T:
        return np.ones((1,K))
    else :
        s = np.einsum('ij,ij->ij',A,beta_rec(t+1, u, mean, covar, A, K, T))
        return np.einsum('ij,lj->li',s,normal_density(np.expand_dims(data[t+1],axis=0), mu, sigma_sq))


#@functools.lru_cache()
#def beta_rec_bis(t, q_t, u, mean, covar, A, K, T):
#    # defined in a recursive form, with vectorization
#    if t == T:
#        return 1.
#    else :
#        s = 0
#        for i in range(K):
#            s += A[q_t,i]*sps.multivariate_normal.pdf(u[t+1], mean=mean[i], 
#                      cov=covar[i])*beta_rec(t+1, i, u, mean, covar, A, K, T)
#        return s


def proba_compute():
    return

#%% Parameters updates

def theta_calc(u, mean, covar, A):
    # suppose that parameters are already 3d matrices (data and mu)
    vec = normal_density_isotrop(data, mu, pi, sigma_sq)
    t = vec/(vec.sum(axis=1))[...,None]
    return t


def pi_calc(p_t):
    return p_t[0]


def A_calc(data, p_t, p_trans):
    return p_trans.sum(axis=0)/(p_t.sum(axis=0))


def mu_calc(data, p_t):
    # suppose that parameters are already 3d matrices (data and mu)
    return (data*p_t).sum(axis=0)/(p_t.sum(axis=0))
    

def sigma_sq_calc_gen(data, mu, tau_ij, *args):
    # suppose that parameters are already 3d matrices (data and mu)
    return np.einsum('ik,ijlk->jlk',tau_ij, np.einsum('ijk,ilk->ijlk',data-mu, data-mu))/tau_ij.sum(axis=0)[None,:]
  

#%% EM algo

def EM_algo(precision, it_max):
    
    # init and reshape variables
    data_res = np.expand_dims(data, axis=2)
    mu = mu_0.T
    mu_res = np.expand_dims(mu, axis=0)
    pi = pi_0
    sigma_sq = sigma_sq_0
    log_like = -float('inf')
    log_like_p = - log_like
    
    # main loop
    it = 0
    while (log_like-log_like_p)**2>precision and it<=it_max:
        
        # E-step (ie tau_ij calculation)
        tau_ij = tau_ij_calc(data_res, pi, mu_res, sigma_sq)

        # M-step
        pi = tau_ij.mean(axis=0)
        tau_ij_res = np.expand_dims(tau_ij, axis=1)
        mu = mu_calc(data_res, tau_ij_res)
        mu_res = np.expand_dims(mu, axis=0)
        sigma_sq = sigma_sq_calc(data_res, mu_res, tau_ij, dim)
        
        log_like_p = log_like
        log_like = log_like_calc(pi, mu_res, sigma_sq, tau_ij)
        
        it += 1
    
    
    
    return 








#%% Main

if __name__ == '__main__':
 
    # address of the file (to be modified) 
    address = '/Users/bastien/Documents/ENS 2018-2019/Probabilistic graphical model/HW3/classification_data_HWK2'
    
    dic_data = pre_process(address)
    train = dic_data['data']
    test = dic_data['test']
    
    K = 4 # number of hidden states
    T_train = len(train)
    
    # initialization
    A = np.ones((K,K))/K # A[i,j] = P(z_t=j|z_(t-1)=i)
    mu_0, sigma_sq_0 = import_init_data() # import mu and sigma_sq found in previous homework
    









