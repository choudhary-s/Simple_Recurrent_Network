import numpy as np
#from rnn_utils import *

def tanh(x):
    return  np.tanh(x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def rnn_cell_back(da_next, cache):
    (at_next, at_prev, xt, parameters) = cache
    waa = parameters["waa"]
    wax = parameters["wax"]
    wya = parameters["wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    dtanh = (1-at_next**2)*da_next
    da_wax = np.dot(dtanh, xt.T)
    da_waa = np.dot(dtanh, at_prev.T)
    da_b   = np.sum(dtanh, axis=1, keepdims=1)
    da_xt  = np.dot(wax.T, dtanh)
    dat_prev = np.dot(waa.T, dtanh)
    
    gradients = {"da_wax":da_wax, "da_waa": da_waa, "da_b":da_b, "da_xt":da_xt, "dat_prev":dat_prev}
    
    return gradients

def rnn_cell(xt,at_prev,parameters):
    waa = parameters["waa"]
    wax = parameters["wax"]
    wya = parameters["wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    at_next = tanh(np.dot(waa,at_prev)+np.dot(wax,xt)+ba)
    yt_predict = softmax(np.dot(wya,at_next)+by)
    
    cache = (at_next, at_prev, xt, parameters)
    return at_next, yt_predict, cache



n_x = 3 
m = 10
n_y = 2
n_a = 5

np.random.seed(1)

xt = np.random.randn(n_x,m)
waa = np.random.randn(n_a,n_a)
wax = np.random.randn(n_a,n_x)
wya = np.random.randn(n_y,n_a)
ba = np.random.randn(n_a,1)
by = np.random.randn(n_y,1)
at_prev = np.random.randn(n_a, m)
da_next = np.random.randn(n_a, m)

parameters = {"waa":waa, "wax":wax, "ba":ba, "by":by, "wya":wya}

a_next, y_next ,cache = rnn_cell(xt,at_prev,parameters) 
gradients = rnn_cell_back(da_next, cache)

print("------------------------I am checking if I am right-------------------------------")
print("a_next.shape ", a_next.shape)
print("y_next.shape ", y_next.shape)
print("a_next[1] ",a_next[1])
print("y_next[4] ",y_next[1])
print("cache length ",len(cache))
print("gradients length", len(gradients))
print("gradients[da_wax].shape", gradients["da_wax"].shape)
print("gradients[da_waa].shape", gradients["da_waa"].shape)
print("gradients[da_b].shape", gradients["da_b"].shape)
print("gradients[da_xt].shape", gradients["da_xt"].shape)
print("gradients[dat_prev].shape", gradients["dat_prev"].shape)

#print("caches ", caches)
