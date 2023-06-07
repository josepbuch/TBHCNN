import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.layers import Dense,Conv2D, MaxPooling2D,Flatten,UpSampling2D,Reshape, BatchNormalization, Activation
from tensorflow.keras.layers import PReLU,LeakyReLU
from tqdm.auto import tqdm

def neural_autoencoder_CNN_PReLU():
	'''
	Definition of the Neural Network, Autoencoder 
	'''
    Net = Sequential()

    # Encoder
    Net.add(Conv2D(input_shape=(20, 20, 3), filters=128, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())

    Net.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())

    Net.add(MaxPooling2D(pool_size=2))

    Net.add(Conv2D(filters=32, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())

    Net.add(MaxPooling2D(pool_size=2))

    # Bottle Neck
    Net.add(Conv2D(filters=16, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())

    # Decoder
    Net.add(UpSampling2D(size=2))

    Net.add(Conv2D(filters=32, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())

    Net.add(UpSampling2D(size=2))

    Net.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())
    
    Net.add(Conv2D(filters=128, kernel_size=(4, 4), padding='same'))
    Net.add(BatchNormalization())
    Net.add(PReLU())

    # Output Layer
    Net.add(Conv2D(filters=3, kernel_size=(1, 1), activation='linear', padding='valid'))

    Net.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    return Net


def training_NN(epoch ,batch, y_input,y_output):

    loss = np.zeros((2,epoch))
    for i in tqdm(range(epoch)):
        history = Net.fit(y_input,y_output,epochs = 1,batch_size = batch,verbose= 1)
        loss[0][i] = history.history['loss'][0]
        loss[1][i] = history.history['accuracy'][0]
    return loss # np.array with the losses and accuracy of the Neural Training




#__________________________________________________________________

# Parameters------------------------------------
shape = 20,20
n1,n2  = shape
nkpts  = n1*n2
epoch = 2500
batch = 100
#-----------------------------------------------

# Data input from the datagenerator
y_input = np.load('')
y_output = np.load('')



Net = neural_autoencoder_CNN_PReLU()
loss = training_NN(epoch ,batch, y_input,y_output)


Net.save('file_path/name_path.keras') # Save the model
np.save('file_path/name_path.npy',loss)# Save the convergence corve


