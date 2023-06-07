import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tqdm.auto import tqdm

#-----------------------------------------------------

DATASET = np.load('file_path/name_file.npy')
parameters = np.load('/file_path/name_file.npy')

Net_autoencoder = tf.keras.models.load_model('file_path/name_file.keras') # Autoencoder Net trained 
Net_atheo = tf.keras.models.load_model('file_path/name_file.npy') # Encoder Net trained

#-----------------------------------------------------

def normalitzador(DATASET):
	'''
	Normalitzate the input data saving the normalitzation parameters
	in kx_parameters and ky_parameters
	'''
    N1,N2 = 20,20
    kx = []
    ky = []
    ep = []
    kx_parameters = [] # lists where min and max will be stored 
    ky_parameters = []
    for i in DATASET:
        sample = i.reshape(400,3)
        sample = sample.T
    
        KX = sample[0]
        kxmin = abs(min(KX))
        kxmax = abs(max(KX))
        
        KY = sample[1]
        kymin = abs(min(KY))
        kymax = abs(max(KY))
        
        kx.append((KX+kxmin)/(kxmax+kxmin))
        ky.append((KY+kymin)/(kymax+kymin))
        ep.append(sample[2])
        kx_parameters.append([kxmin,kxmax])
        ky_parameters.append([kymin,kymax])
    
    kx = np.array(kx).reshape(len(parameters),N1,N2)
    ky = np.array(ky).reshape(len(parameters),N1,N2)
    ep = np.array(ep).reshape(len(parameters),N1,N2)

    y_input = np.stack((np.array(kx), np.array(ky), np.array(ep)),axis = 3 )   
         
    return y_input, kx_parameters, ky_parameters


def desnorm(y_denoised, kx_param,ky_param):
	'''
	Desnormalitzate the list with the parameters of the normalitzation function
	'''
    N1,N2 = 20,20
    kx = []
    ky = []
    ep = []
    for i,j,k in zip(y_denoised,kx_param,ky_param):
        sample = i.reshape(400,3)
        sample = sample.T
    
        KX = sample[0]
        kxmin = j[0]
        kxmax = j[1]
        
        KY = sample[1]
        kymin = k[0]
        kymax = k[1]
        
        kx.append(KX*(kxmin + kxmax)- kxmin)
        ky.append(KY*(kymin + kymax)- kymin)
        ep.append(sample[2])

    
    kx = np.array(kx).reshape(len(parameters),N1,N2)
    ky = np.array(ky).reshape(len(parameters),N1,N2)
    ep = np.array(ep).reshape(len(parameters),N1,N2)

    y_final = np.stack((np.array(kx), np.array(ky), np.array(ep)),axis = 3 )   
         
    return y_final



# Executation of the program 

#Normalitzate the data 
y_input, kx_param, ky_param = normalitzador(DATASET)
# Denoise the image
y_denoised = Net_autoencoder.predict(y_input)
# Desnormalitzate
y_final = desnorm(y_denoised, kx_param,ky_param)
#Final parameter calculation 
params_final = Net_atheo.predict(y_final)


# Accuracy of the Model respect the parameters 

q = np.divide(params_final, parameters)
err_rel = np.abs(q-1)
errors = []
for i in err_rel.T:
    errors.append((np.sum(i)/len(parameters)))
print ('error a:',errors[0]*100, '%')
print ('error t:', errors[1]*100, '%')
print ('error e0:', errors[2]*100, '%')