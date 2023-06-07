import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Activation,Flatten,MaxPooling2D,Dropout
from tqdm.auto import tqdm

def neural_conv(loss_func,optimizer_fun,activation_fun):
	'''
	Defined the NN Encoder with the parameters for the tranining
	Input : loss function, Activation fuction, optimizer
	Output: Net 
	'''


    Net = Sequential()

    Net.add(Conv2D(input_shape = (20,20,3), filters = 64, kernel_size = [4,4], padding = 'same'))
    Net.add(BatchNormalization())
    Net.add(Activation(activation_fun))
    
    Net.add(MaxPooling2D(pool_size = 2))
    
    Net.add(Conv2D(filters = 32, kernel_size = [3,3], padding = 'same'))
    Net.add(BatchNormalization())
    Net.add(Activation(activation_fun))

    Net.add(MaxPooling2D(pool_size = 2))

    Net.add(Conv2D(filters = 16, kernel_size = [2,2], padding = 'same'))
    Net.add(BatchNormalization())
    Net.add(Activation(activation_fun))

    Net.add(Conv2D(filters = 8, kernel_size = [2,2], padding = 'same'))
    Net.add(BatchNormalization())
    Net.add(Activation(activation_fun))
    
    Net.add(Flatten())
    Net.add(Dense(3, activation = 'linear'))
    
    Net.compile(loss = loss_func, optimizer = optimizer_fun, metrics = ['accuracy'])
    
    return Net 


def training_NN(epoch ,batch, y_input, parameters):
	'''
	Training Subrutine with dataset for the training
	'''

    loss = np.zeros((2,epoch))
    for i in tqdm(range(epoch)):
        history = Net.fit(y_input,parameters,epochs = 1,batch_size = batch,verbose= 0)
        loss[0][i] = history.history['loss'][0]
        loss[1][i] = history.history['accuracy'][0]
        
    return loss # np.array with the losses and accuracy of the Neural Training



# Parameters------------------------------------
shape = 20,20
n1,n2  = shape
nkpts  = n1*n2
nsample = 20000
epoch = 200
batch = 100
#-----------------------------------------------

# Data for the training
DATASET = np.load('file_path')
parameters = np.load('file_path')

loss_func = 'mse'
optimizer = 'adam'
activate_func = 'relu'
Net = neural_conv(loss_func,optimizer,activate_func)

loss = training_NN(epoch ,batch, DATASET, parameters)

Net.save('file_path/Encoder.keras') # Save Net trained
np.save('file_path/Encoder.npy',loss)

    