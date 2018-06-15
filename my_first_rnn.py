import numpy as np
#from rnn_utils import *

def tanh(x):
    return  np.tanh(x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

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

def rnn_cell_forward(x,a_prev,parameters):
    n_y, n_a = parameters["wya"].shape
    n_x, m, T_x = x.shape
    
    a = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    at_next = a_prev
    
    caches = []
    
    for i in range(T_x):
        at_next, yt_predict, cache = rnn_cell(x[:,:,i],at_next,parameters)
        a[:,:,i] = at_next
        y[:,:,i] = yt_predict
        caches.append(cache)
    cache = (cache, x)
    return a, y, caches

n_x = 3 
m = 10
n_y = 2
n_a = 5
T_x = 4

np.random.seed(1)

x = np.random.randn(n_x,m,T_x)
waa = np.random.randn(n_a,n_a)
wax = np.random.randn(n_a,n_x)
wya = np.random.randn(n_y,n_a)
ba = np.random.randn(n_a,1)
by = np.random.randn(n_y,1)
at_prev = np.random.randn(n_a,m)

parameters = {"waa":waa, "wax":wax, "ba":ba, "by":by, "wya":wya}

a,y,caches = rnn_cell_forward(x,at_prev,parameters) 

print("------------------------I am checking if I am right-------------------------------")
print("a.shape ", a.shape)
print("y.shape ", y.shape)
print("a[4][1] ",a[4][1])
print("y[1][4] ",y[1][4])
print("cache length ",len(caches))
#print("caches ", caches)
