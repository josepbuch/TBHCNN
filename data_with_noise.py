import numpy as np 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def energy(a,t,e0,kx,ky,sign):
    _f = t*t*(3+4*np.cos((3/2)*a*kx)*np.cos(((np.sqrt(3)/2))*a*ky)+ 2*np.cos(np.sqrt(3)*a*ky))
    return e0 +sign*(_f**2-2*e0*_f)


def new_dataset(shape, parameters,energy,noise):
    """
    generate de dataset input the parameters and the shape of the data, output 
    valors of kx and ky, and Energies positives and negatives 
    """
    N1,N2 = shape
    n1   = np.linspace(0,1,N1,endpoint=False)
    n2   = np.linspace(0,1,N2,endpoint=False)
    x,y  = [ x.flatten() for x in np.meshgrid(n1, n2) ] 
    grid = np.array([x,y]).T 

    kx = []
    ky = []
    ep = []
    ep_noise = []
    for p in tqdm(parameters):
    
        a = p[0]
        t = p[1]
        e0 = p[2]

        A = (a/2)*np.array( [[3, 3], [np.sqrt(3), -np.sqrt(3)]]  ) # The lattice vectors as COLUMNS
        B = np.linalg.inv(A).T * 2*np.pi # The reciprocal lattice vectors as columns
        kgrid= grid.dot(B.T)
        
        # Normalitzate the kx and ky
        KX = kgrid.T[0]
        kxmin = abs(min(KX))
        kxmax = abs(max(KX))
        KY = kgrid.T[1]
        kymin = abs(min(KY))
        kymax = abs(max(KY))
        
        kx.append((KX+kxmin)/(kxmax+kxmin))
        ky.append((KY+kymin)/(kymax+kymin))
        
        # Add noise
        energia = energy(a,t,e0,KX,KY, +1)
        
        random = np.random.uniform(-noise,noise, (N1*N2))
        energy_noise = energia*random + energia
        
        ep.append(energia)
        ep_noise.append(energy_noise)
        
    # Reshape the data 
    kx = np.array(kx).reshape(len(parameters),N1,N2)
    ky = np.array(ky).reshape(len(parameters),N1,N2)
    ep = np.array(ep).reshape(len(parameters),N1,N2)
    ep_noise = np.array(ep_noise).reshape(len(parameters),N1,N2)

    y_output = np.stack((np.array(kx), np.array(ky), np.array(ep)),axis = 3 )   
    y_input = np.stack((np.array(kx), np.array(ky), np.array(ep_noise)),axis = 3 )   

    return y_input, y_output 




# Parameters------------------------------------
shape = 20,20
n1,n2  = shape
nkpts  = n1*n2
nsample = 50000
noise = 0.1
#-----------------------------------------------
parameters = 3.5*np.random.rand(nsample,3) + 0.5
y_input,y_output =  new_dataset(shape,parameters,energy, noise)

np.save('file_path/file_name.npy',y_output) # data with Noise
np.save('file_path/file_name.npy', y_input) # data without Noise 
np.save('file_path/file_name.npy', parameters)
