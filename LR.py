#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading the train_data_file.txt file 
import pandas as pd 
data = pd. read_csv('train_data_file.txt', sep=" ",header=None)
# print(data)
#splitting data into x(input), y(output)
x = data[0] 
y= data[1]
# print (x)
# print (y) 


# In[2]:



#plotting training data
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[3]:


#constructing the model of linear regression y=Wx+b that is a single layer using keras library 
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
#creating model
model = Sequential()
model.add(Dense(1, input_dim=1))# 1 layer 1 Dimension
#optimizing the loss function of the mean square using adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()#to show the model and check the no. of layers and input parameters


# In[5]:

# Training data giving input and label to the model to output the function
Training_model = model.fit(x,y,epochs=60)
#the first time I used 100 epoches the loss was nearly equal to 0 but after seeing the loss-epoches graph
# I saw that using 60 epoches was a good compromise for error percentage and time and to avoid overfitting


#training model setting epoches with 100 , and taking 20% of the training data to validation data 
# Training_model = model.fit(x,y,validation_split=0.2, epochs=100) 


# In[6]:


# plotting the lossfunction to see the best number for epoches withrespect to epoches

plt.plot(Training_model.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoches')
plt.show()

#From the graph we should choose nearly 40 to 60 epoches to save power and time and have good error percentage
#as we can see the loss doesn't get that much reduced by increasing the epoches and to avoid overfitting too

# In[12]:


#plotting Training data line error vs Validation data line error
# plt.plot(Training_model.history['loss'],Training_model.history['val_loss'])
# plt.ylabel('validation data error')
# plt.xlabel('training data error')
# plt.show()


# In[8]:


#testing the model (with our input data)
test= model.predict([x])

#plotting output of predicted vs original data 
#predicted line in red circle marker
#original line in blue 
plt.plot(x,test ,'ro')
plt.plot(x,y ,'b')
plt.ylabel('output')
plt.xlabel('input')
plt.legend(['test', 'train'], loc='upper left')
plt.show()


# In[11]:


#defining a function for prediction
def LR_predict(x):
	return model.predict([x])

#predicting a random value input
LR_predict(10005)


# In[ ]:




