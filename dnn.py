import numpy as np

def initialize_parameters(layer_dims):
    parameters = {}
    layer_num = len(layer_dims)
    for l in range (1, layer_num) :
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) # *0.01 is not a good parameter
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def softmax(Z):
    A = np.exp(Z)/np.sum(np.exp(Z), axis=0, keepdims=True)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s, _ = sigmoid(Z)
    dZ = dA * s * (1-s) # dA * derivative of sigmoid function
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # make a copy of dA
    # relu function is max(0,Z), so when Z <= 0, the derivative of relu is
    # 1, when Z > 0, 0 when Z <= 0
    # so dA * derivative of relu equals dA when Z > 0, 0 when Z <= 0
    dZ[Z <= 0] = 0
    return dZ

def compute_forward(A, W, b, activation):
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    activation_cache = []
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    if activation == "relu":
        A, activation_cache = relu(Z)
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation(X, parameters, activation_dims):
    caches = []
    A = X
    layer_num = len(parameters) // 2 # parameters include W and b, so len // 2 is layer num
    for l in range(1, layer_num):
        A_prev = A
        A, cache = compute_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation=activation_dims[l-1])
        caches.append(cache)
    A, cache = compute_forward(A, parameters['W'+str(layer_num)], parameters['b'+str(layer_num)], activation=activation_dims[layer_num])
    caches.append(cache)

    return A, caches

def compute_cost(A_pred, Y, out_activation):
    # compute cost with cross-entropy
    m = Y.shape[1]
    cost = 0
    if out_activation == "sigmoid":
        cost = (1./m) * (-np.dot(Y,np.log(A_pred).T) - np.dot(1-Y, np.log(1-A_pred).T))
    if out_activation == "softmax":
        cost = (1./m) * (-np.sum(Y * np.log(A_pred)))
    cost = np.squeeze(cost) # to make sure the cost shape is a number
    return cost

def compute_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    dZ = []
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    if activation == "softmax":
    # The derivative of the cost of softmax function with respect to Z can be computed as
    # dC/dz = sum(dC/dy * dy/dz) = A_pred - Y
    # This is easier than compute dA_pred and then compute dZ
        dZ = dA
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward_propagation(A_pred, Y, caches, activation_dims):
    grads = {}
    dA_pred = []
    layer_num = len(caches)
    m = A_pred.shape[1]
    Y = Y.reshape(A_pred.shape) # make sure the shape of Y is same as shape of A_pred
    # derivative of cost with respect to A_pred
    # cost = 1/m*(-Y*log(A_pred).T-(1-Y)*log(1-A_pred).T)
    # dcost/dA_pred = - (Y/A_pred - (1-Y)/(1-A_pred))
    if activation_dims[-1] == "sigmoid":
        dA_pred = - (np.divide(Y, A_pred) - np.divide(1 - Y, 1 - A_pred))
    if activation_dims[-1] == "softmax":
        # at this point, dA_pred is dZ
        dA_pred = A_pred-Y
    current_cache = caches[layer_num-1]
    grads["dA"+str(layer_num)], grads["dW"+str(layer_num)], grads["db"+str(layer_num)] = compute_backward(dA_pred, current_cache, activation_dims[-1])
    # We don't need to compute dA for input layer because in supervised learning, we shouldn't change input.
    # So, we compute dA1, dW1, db1 to dA(l-1), dW(l-1), db(l-1) in the loop
    for l in reversed(range(layer_num - 1)):
        current_cache = caches[l]
        grads["dA"+str(l + 1)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = compute_backward(grads["dA"+str(l+2)], current_cache, activation_dims[l])
    return grads

def update_parameters(parameters, grads, learning_rate):
    layer_num = len(parameters) // 2
    for l in range(layer_num):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
    return parameters

def predict(X, parameters):
    m = X.shape[1]
    layer_num = len(parameters)
    pred, _ = forward_propagation(X, parameters)
    return pred

if __name__ == '__main__':
    pass