import numpy as np

def tanh(x):
    return np.tanh(x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def rnn_one_cell_backprop(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    waa = parameters["waa"]
    wax = parameters["wax"]
    wya = parameters["wya"]
    ba  = parameters["ba"]
    by  = parameters["by"]
    
    dtanh = (1-a_next**2)*da_next
    da_wax = np.dot(dtanh, xt.T)
    da_waa = np.dot(dtanh, a_prev.T)
    da_b = np.sum(dtanh, axis=1, keepdims=1)
    da_xt = np.dot(wax.T, dtanh)
    da_aprev = np.dot(waa.T, dtanh)
    
    gradients = {"da_wax":da_wax, "da_waa":da_waa, "da_b":da_b, "da_xt":da_xt, "da_aprev":da_aprev}
    return gradients

def rnn_backwards(da, caches):
    (cache, x) = caches
    (a_next, a_prev, xt, parameters) = cache
    
    nx, m, Tx = x.shape
    na, m, Tx = da.shape
    
    
    dwax = np.zeros((na, nx))
    dwaa = np.zeros((na, na))
    db   = np.zeros((na, 1))
    dx   = np.zeros((nx, m, Tx))
    da0  = np.zeros((na, m))
    dprev_  = np.zeros((na, m))
    
    
    for i in reversed(range(Tx)):
        gradients = rnn_one_cell_backprop(da[:,:,i]+dprev_, cache[i])
        dwax_, dwaa_, db_, dx_, dprev_ = gradients["da_wax"], gradients["da_waa"], gradients["da_b"], gradients["da_xt"], gradients["da_aprev"]
        dwax+=dwax_
        dwaa+=dwaa_
        db+=db_
        dx[:,:,i] = dx_
    da0 = dprev_
    
    gradients = {"dwax":dwax, "dwaa":dwaa, "db":db, "dx":dx, "da0":da0}
    return gradients


def rnn_cell(xt, a0, parameters):
    waa = parameters["waa"]
    wax = parameters["wax"]
    wya = parameters["wya"]
    ba  = parameters["ba"]
    by  = parameters["by"]
    
    a_next = tanh(np.dot(wax,xt)+np.dot(waa,a0)+ba)
    y_next = softmax(np.dot(wya, a_next)+by)
    
    cache = (a_next, a0, xt, parameters)
    return a_next, y_next, cache

def rnn_forward(x, a0, parameters):
    waa = parameters["waa"]
    wax = parameters["wax"]
    wya = parameters["wya"]
    ba  = parameters["ba"]
    by  = parameters["by"]
    
    nx, m, Tx = x.shape
    ny, na = wya.shape
    
    a_next = a0
    a = np.zeros((na, m, Tx))
    y = np.zeros((ny, m, Tx))
    
    caches = []
    
    for i in range(Tx):
        a_next, y_next, cache = rnn_cell(x[:,:,i], a_next, parameters)
        a[:,:,i] = a_next
        y[:,:,i] = y_next
        caches.append(cache)
    
    caches = (caches, x)
    
    return a, y, caches

na = 5
nx = 3
m  = 10
ny = 2
Tx = 4

np.random.seed(1)
waa = np.random.randn(na,na)
wax = np.random.randn(na,nx)
wya = np.random.randn(ny,na)
ba  = np.random.randn(na, 1)
by  = np.random.randn(ny, 1)
a0  = np.random.randn(na, m)
x  = np.random.randn(nx, m, Tx)
da = np.random.randn(na, m, Tx)

parameters = {"waa":waa, "wax":wax, "wya":wya, "ba":ba, "by":by}

a, y, caches = rnn_forward(x, a0, parameters)
gradients = rnn_backwards(da, caches)
#(cache, x) = caches
#gradients = rnn_one_cell_backprop(da_next, cache[0])
print("-------------------------I am just checking-------------------------------")
print("a.shape", a.shape)
print("y.shape", y.shape)
print("a[4][1]", a[4][1])
print("y[1][4]", y[1][4])
print("gradients length", len(gradients))
print("cache length", len(caches))
print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dwax\"][3][1] =", gradients["dwax"][3][1])
print("gradients[\"dwax\"].shape =", gradients["dwax"].shape)
print("gradients[\"dwaa\"][1][2] =", gradients["dwaa"][1][2])
print("gradients[\"dwaa\"].shape =", gradients["dwaa"].shape)
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"db\"].shape =", gradients["db"].shape)