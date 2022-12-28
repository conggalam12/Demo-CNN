import numpy as np
import pandas as pb
def sigmoid(x):
    w = 1/(1+np.exp(-x))
    return w
def sigmoid_backward(x):
    w = 1/(1+np.exp(-x))
    return w*(1-w)
def ReLu(x):
    return np.maximum(0,x)
def ReLu_backward(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x
def linear(w,x,b):
    z = np.dot(w,x)+b
    return z
def tanh(x):
    w = np.exp(x)-np.exp(-x)
    a = np.exp(x)+np.exp(-x)
    return w/a
def tanh_backward(x):
    a = np.exp(x)+np.exp(-x)
    return 4/(a*a)
def function(x,name):
    if name =='tanh':
        return tanh(x)
    elif name =='sigmoid':
        return sigmoid(x)
    elif name =='relu':
        return ReLu(x)
    return 0
def derivative(x,name):
    if name =='tanh':
        return tanh_backward(x)
    elif name =='sigmoid':
        return sigmoid_backward(x)
    elif name =='relu':
        return ReLu_backward(x)
    return 0
#demo layer [2,2,1] : n[0] = 2, n[1] = 2 , n[2] = 1
class Neural_Network:
    def __init__(self,layers,learning_rate,activate_function):
        self.layers = layers 
        self.learning_rate = learning_rate
        self.caches = {}
        self.activate_function = activate_function
   
    def create_parameters(self):
        for i in range(0,len(self.layers)-1):
            w = np.random.randn(self.layers[i+1],self.layers[i])
            b = np.zeros((self.layers[i+1],1))
            self.caches['W'+str(i+1)] = w
            self.caches['b'+str(i+1)] = b

    
    def train(self,x,y):
        # forward
        A=[x]
        Z=[x]
        a = x
        for i in range(0,len(self.layers)-1):
            z = linear(self.caches['W'+str(i+1)],a,self.caches['b'+str(i+1)])
            f = self.activate_function[i]
            a = function(z,f)
            A.append(a)
            Z.append(z)
        #backpropagation
        y=y.reshape(-1,1)
        dA = [-(y/A[-1]-(1-y)/(1-A[-1]))] # sigmoid 
        dW = []
        db = []
        for i in reversed(range(0,len(self.layers)-1)):
            t = derivative(Z[i+1],self.activate_function[i])
            dW_ = np.dot(dA[-1]*t,(A[i]).T)
            db_ = (np.sum(dA[-1]*t,0)).reshape(-1,1)
            dA_ = np.dot(dA[-1]*t,self.caches['W'+str(i)].T)
            dW.append(dW_)
            db.append(db_)
            dA.append(dA_)
        
        #reversed
        dW = dW[::-1]
        db = db[::-1]

        # SGD

        for i in range(0,len(self.layers)-1):
            self.caches['W'+str(i)] = self.caches['W'+str(i)] - self.learning_rate*dW[i]
            self.caches['b'+str(i)] = self.caches['b'+str(i)] - self.learning_rate*db[i]
    def fit(self,x,y,epochs=5):
        for epoch in range(0,epochs):
            self.train(x,y)
            loss = self.loss_function(x,y)
            print("Epoch {} , loss {}".format(epoch+1,loss))
    def predict(self,x):
        for i in range(0,len(self.layers)-1):
            l = linear(x,self.caches['W'+str(i)],self.caches['b'+str(i)])
            x = sigmoid(l)
        return x
    def loss_function(self,x,y):
        y_hat = self.predict(x)
        loss = np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
        return -loss

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([0,1,1])
f = ['relu','sigmoid']
layer = [3,2,1]
demo = Neural_Network(layer,0.01,f)
demo.create_parameters()
demo.train(x,y)
        