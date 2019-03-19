
# coding: utf-8

# # math 287 code

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
from model import *


# In[15]:


from sklearn.decomposition import SparsePCA
import pickle
from sklearn.decomposition import PCA


# # handwritting data

# In[18]:


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm


# In[19]:


def load_data(fname):
    def one_hot(target):
        N = len(target)
        T = np.zeros((N, 10), dtype=int)
        for i in range(N):
            T[i][target[i]] = 1
        return T  # shape (# samples, # classes)

    images=[]
    labels=np.zeros(10,dtype = int)
    with open(fname, 'rb') as f:
        images = pickle.load(f)
    labels = images[:,-1]
    labels = [int(i) for i in labels]
    images = np.delete(images, -1, axis=1)
    labels = one_hot(labels)
    return images, labels


# In[20]:


images,labels = load_data('C:/Users/Yang Xu/Documents/PA2-Backprop-master/data/MNIST_test.pkl')
images.shape


# In[21]:


images_new,labels_new = images[:100],labels[:100]
images_new = images_new - np.mean(images_new)


# In[22]:


transformer_spca = SparsePCA(n_components=50,
         random_state=None,method='lars',max_iter=100)


# In[23]:


transformer_spca.fit(images_new)


# In[24]:


images_spca = transformer_spca.transform(images_new)
images_spca.shape


# In[25]:


transformer_pca = PCA(n_components=50) # n_components == min(n_samples, n_features)
transformer_pca.fit(images_new)
images_pca = transformer_pca.transform(images_new)


# In[26]:


images_pca.shape


# In[27]:


loadings = transformer_pca.components_
loadings_spca = transformer_spca.components_


# In[33]:


def plot_digit(n):
    fig = plt.figure()
    for i in range(n):
        img1 = loadings_spca[i+1,:].reshape((28,28))
        ax = fig.add_subplot(7, 7, i + 1)
        plt.axis('off')
        ax.imshow(img1, cmap=plt.cm.bone)
        
    plt.savefig(f"C:/Users/Yang Xu/Documents/PA2-Backprop-master/img/pca_loading.png")
    plt.show()


# In[34]:


plot_digit(49)


# In[47]:


if __name__ == "__main__":
    train_data_fname = 'C:/Users/Yang Xu/Documents/PA2-Backprop-master/data/MNIST_train.pkl'
    valid_data_fname = 'C:/Users/Yang Xu/Documents/PA2-Backprop-master/data/MNIST_valid.pkl'
    test_data_fname = 'C:/Users/Yang Xu/Documents/PA2-Backprop-master/data/MNIST_test.pkl'


# In[48]:


def line_width(ep):
    return min(max(1.65 - ep / 700, 0.8), 1.5)


# In[152]:


def plot_acc(result_dict, epochs_after_optim=0, config=None, description=None, adjust_linewidth=True):
    # Initialize the figure
    my_dpi = 120
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    plt.style.use('ggplot')

    # A color palette
    palette = plt.get_cmap('Set1')

    # Number of epochs to plot
    ep = len(result_dict['valid_loss'])
    if epochs_after_optim is None:
        result = result_dict.copy()
    else:
        ep = min(1 + np.argmin(result_dict['valid_loss']) + epochs_after_optim, ep)
        result = {k: v[:ep] for k, v in result_dict.items()}

    # Plots
    subplots = [['Classification Loss (Cross-Entropy)', 'train_loss', 'valid_loss']]
    for s, subplot in enumerate(subplots):
        plt.subplot(1, 1, s + 1)

        title, key1, key2 = subplot
        lw = line_width(ep) if adjust_linewidth else 1.3

        plt.plot(range(ep), result[key1], marker='', color=palette(0), lw=lw, alpha=0.7)
        plt.plot(range(ep), result[key2], marker='', color=palette(1), lw=lw, alpha=0.7)
        plt.legend(['train', 'valid'])  # , prop={'size': 8})
        plt.title(title, loc='center', style='italic', fontsize=12, fontweight=0)

    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)  # Reset


# In[131]:


### Train the network ###
model = Neuralnetwork(config)
X_train, y_train = load_data(train_data_fname)
X_valid, y_valid = load_data(valid_data_fname)
X_test, y_test = load_data(test_data_fname)
trainer(model, X_train, y_train, X_valid, y_valid, config)
test_acc = test(model, X_test, y_test, config)


# In[133]:


raw = X_train,y_train,X_valid, y_valid,X_test, y_test 
#X_train,y_train,X_valid, y_valid,X_test, y_test = raw


# In[134]:


X_train,y_train,X_valid, y_valid,X_test, y_test = X_train[:200],y_train[:200],X_valid[:200],y_valid[:200],X_test[:200],y_test[:200]


# # perform SPCA

# In[57]:


transformer_spca = SparsePCA(n_components=50,
         random_state=None,method='lars',max_iter=1000)
transformer_spca.fit(X_train)


# In[58]:


X_train_spca = transformer_spca.transform(X_train)
X_train_spca.shape


# In[59]:


transformer_spca.fit(X_valid)
X_valid_spca = transformer_spca.transform(X_valid)
X_valid_spca.shape


# In[60]:


transformer_spca.fit(X_test)
X_test_spca = transformer_spca.transform(X_test)
X_test_spca.shape


# # Perform PCA

# In[61]:


transformer_pca = PCA(n_components=50) # n_components == min(n_samples, n_features)
transformer_pca.fit(X_train)
X_train_pca = transformer_pca.transform(X_train)
transformer_pca.fit(X_valid)
X_valid_pca = transformer_pca.transform(X_valid)
transformer_pca.fit(X_test)
X_test_pca = transformer_pca.transform(X_test)


# # Train

# In[161]:


# Configuration for part (c)
config_c = config.copy()
config_c['layer_specs'] = [50, 15, 10]
config_c['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config_c['epochs'] = 300
config_c['learning_rate'] = 0.0006
config_c['L2_penalty'] = 0.0001 # lambda = 0.001
config_c['early_stop'] = True
print("The current configuration:\n", config_c)
# Training and testing
model = Neuralnetwork(config_c)
train_result = trainer(model, X_train_pca, y_train, X_valid_pca, y_valid, config_c)


# In[162]:


plot_acc(train_result, 0, config_c, 'part_c',adjust_linewidth=True)


# ### Configuration for part (c)
# config_c = config.copy()
# config_c['momentum'] = False
# config_c['layer_specs'] = [50, 40, 10]
# config_c['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
# config_c['epochs'] = 200
# config_c['learning_rate'] = 0.0006
# config_c['L2_penalty'] = 0.0001 # lambda = 0.001
# config_c['early_stop'] = False
# print("The current configuration:\n", config_c)
# # Training and testing
# model = Neuralnetwork(config_c)
# #train_result = trainer(model, X_train_spca, y_train, X_valid_spca, y_valid, config_c)
# train_result = trainer(model, X_train_spca, y_train, X_valid_spca, y_valid, config_c)
# test_acc = test(model, X_test_spca, y_test, config_c)

# In[163]:


# Configuration for part (c)
config_c = config.copy()
config_c['momentum'] = False
config_c['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config_c['epochs'] = 300
config_c['layer_specs'] = [50, 15, 10]
config_c['learning_rate'] = 0.0006
config_c['L2_penalty'] = 0.0001 # lambda = 0.001
config_c['early_stop'] = True
print("The current configuration:\n", config_c)

# for i in [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]:
#     config_c['layer_specs'] = [50, i, 10]

# Training and testing
model = Neuralnetwork(config_c)
#train_result = trainer(model, X_train_spca, y_train, X_valid_spca, y_valid, config_c)
train_result = trainer(model, X_train_spca, y_train, X_valid_spca, y_valid, config_c)


# In[164]:


plot_acc(train_result, None, config_c, 'part_c',adjust_linewidth=True)

