
# coding: utf-8

# In[1]:


import numpy as np
import PIL
import matplotlib.pyplot as plt
import os
import random
from os import listdir
from PIL import Image
import numpy as np


# In[2]:


from sklearn import datasets
import pandas as pd


# In[3]:


import json
import matplotlib as mpl
from jupyterthemes import jtplot
import sys

# In[15]:


from sklearn.decomposition import SparsePCA
import pickle
from sklearn.decomposition import PCA




def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    # here, x and y are both np arrays
    #output = np.exp(x)/sum(np.exp(x))
    y = np.exp(x)
    output = y/y.sum(axis=1)[:,None]
    output = np.maximum(output,sys.float_info.min)
    return output
def accuracy(outputs, targets):
    """
    Computing accuracy given outputs (predictions) and targets.
    Takes in arrays and return a float number.
    """
    correct = np.argmax(outputs, axis=1) == np.argmax(targets, axis=1) # max along the col
    return sum(correct) / len(correct)


# In[41]:


class Activation:
    def __init__(self, activation_type = "sigmoid"):
        self.activation_type = activation_type
        self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)
    
        elif self.activation_type == "tanh":
            return self.tanh(a)
    
        elif self.activation_type == "ReLU":
            return self.ReLU(a)
  
    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()
    
        elif self.activation_type == "tanh":
            grad = self.grad_tanh()
    
        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
    
        return grad * delta
      
    def sigmoid(self, x):
        """
        Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = 1/(1+ np.exp(-x))
        return output

    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = np.tanh(x)
        return output

    def ReLU(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        #output = x if x > 0 else 0
        output = (x > 0) * x
        return output

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        grad = np.exp(self.x)/(1+np.exp(self.x))**2
        return grad

    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        return 1 - np.tanh(self.x)**2

    def grad_ReLU(self):
        """
        Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        grad = (self.x > 0).astype(np.float32)
        return grad


# In[42]:


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.x = x
        self.a = np.dot(np.ones((x.shape[0], 1)), self.b) + np.dot(x, self.w)
        return self.a #matrix
  
    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        self.d_x = np.dot(delta, self.w.T)
        self.d_w = np.dot(self.x.T, delta)
        self.d_b = delta.sum(axis=0)[None, :]   # or: np.dot(np.ones((1, self.x.shape[0])), delta)
        return self.d_x


# In[43]:


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))  
    
    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        # Through all layers
        for layer in self.layers:
            x = layer.forward_pass(x)

        # Computing the outputs
        self.y = softmax(x)

        # Computing the loss
        if targets is None:
            loss = None
        else:
            self.targets = targets
            loss = self.loss_func(np.log(self.y), targets)

        return loss, self.y

    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        shape = targets.shape
        N, C = shape if len(shape) > 1 else (1, shape)
        return -(targets * logits).sum() / N / C
    
    def backward_pass(self):
        '''
        implement the backward pass for the whole network. 
        hint - use previously built functions.
        '''
        grad = self.targets - self.y
        for layer in self.layers[::-1]:
            grad = layer.backward_pass(grad)


# In[124]:


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """
    print("\nTraining...")

    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    alpha, lam = config['learning_rate'], config['L2_penalty']

    if config['early_stop']:
        # Save the weights and biases after each epoch
        weights = [[(layer.w, layer.b) if isinstance(layer, Layer) else None for layer in model.layers]]
        count = 0  # (Initial) count of epochs that validate loss keeps to increase

    # Initial train loss, acc
    loss, outputs = model.forward_pass(X_train, y_train)
    train_loss.append(loss)
    train_acc.append(accuracy(outputs, y_train))

    # Initial valid loss, acc
    loss, outputs = model.forward_pass(X_valid, y_valid)
    valid_loss.append(loss)
    valid_acc.append(accuracy(outputs, y_valid))

    # Mini-batches
    num_batch = int(np.ceil(len(y_train) / config['batch_size']))
    for epoch in range(config['epochs']):
        train_idx = np.arange(len(y_train))
        random.shuffle(train_idx)

        for m in range(num_batch):
            idx = train_idx[m * config['batch_size']:(m + 1) * config['batch_size']]
            X = X_train[idx, :]  # inputs
            T = y_train[idx, :]  # targets

            # Forward pass and backward pass through all layers
            model.forward_pass(X, T)
            model.backward_pass()

            # Update the weights and biases
            for layer in model.layers:
                if isinstance(layer, Layer):
                    layer.w += alpha * layer.d_w - alpha * lam * layer.w
                    layer.b += alpha * layer.d_b - alpha * lam * layer.b

        # Train loss, acc
        loss, outputs = model.forward_pass(X_train, y_train)
        train_loss.append(loss)
        train_acc.append(accuracy(outputs, y_train))

        # Valid loss, acc
        loss, outputs = model.forward_pass(X_valid, y_valid)
        valid_loss.append(loss)
        valid_acc.append(accuracy(outputs, y_valid))

        # Early stopping
        if config['early_stop']:
            # Save the weights and biases
            w = [(layer.w, layer.b) if isinstance(layer, Layer) else None for layer in model.layers]
            weights.append(w)
            count += 1 if valid_loss[-1] > valid_loss[-2] else -count

            # The epoch with minimum validate loss
            best_epoch = np.argmin(valid_loss)

            if count >= config['early_stop_epoch']:
                # Recover the weights/biases for hidden and input layers
                for i, layer in enumerate(model.layers):
                    if isinstance(layer, Layer):
                        layer.w, layer.b = weights[best_epoch][i]
                print(f"Stopped at the {epoch + 1}-th epoch.")
                print(f"Minimum validation loss at the {best_epoch}-th epoch.")
                break

            if epoch == config['epochs'] - 1:
                print(f"Reached the maximum number of epochs ({epoch + 1}).")
                print(f"Minimum validation loss at the {best_epoch}-th epoch.")

        elif epoch == config['epochs'] - 1:
            print("Done.")
    print(min(valid_loss))

    return {"train_loss": train_loss, "train_acc": train_acc,
            "valid_loss": valid_loss, "valid_acc": valid_acc}


# In[125]:


def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    print("\nEvaluating on the test set...")

    _, outputs = model.forward_pass(X_test, y_test)
    acc = accuracy(outputs, y_test)

    print(f"Test accuracy {acc * 100:.2f}%.")
    return acc
