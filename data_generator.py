import numpy as np 
from tqdm.auto import tqdm


def energy(a,t,e0,kx,ky,sign):
    _f = t*t*(3+4*np.cos((3/2)*a*kx)*np.cos(((np.sqrt(3)/2))*a*ky)+ 2*np.cos(np.sqrt(3)*a*ky))
    return e0 +sign*(_f**2-2*e0*_f)


def new_dataset(shape, parameters,energy):
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
    for p in tqdm(parameters):
    
        a = p[0]
        t = p[1]
        e0 = p[2]

        A = (a/2)*np.array( [[3, 3], [np.sqrt(3), -np.sqrt(3)]]  ) # The lattice vectors as COLUMNS
        B = np.linalg.inv(A).T * 2*np.pi # The reciprocal lattice vectors as columns
        kgrid= grid.dot(B.T)
        
        # Taking the energies
        kx.append(kgrid.T[0])
        ky.append(kgrid.T[1])
        ep.append(energy(a,t,e0,kgrid.T[0],kgrid.T[1], +1))
        
    # Reshape the data 
    kx = np.array(kx).reshape(len(parameters),N1,N2)
    ky = np.array(ky).reshape(len(parameters),N1,N2)
    ep = np.array(ep).reshape(len(parameters),N1,N2)
    dataset = np.stack((np.array(kx), np.array(ky), np.array(ep)),axis = 3 )   
        
    return dataset # Returns an array of (len(parameters),N1,N2,3) shape 


#_________________________________________________________________

# Parameters------------------------------------
shape = 20,20
n1,n2  = shape
nkpts  = n1*n2
nsample = 20000
#-----------------------------------------------

# creating parameters and DATASET, and save it in .npy documents
parameters = 3.5*np.random.rand(nsample,3) + 0.5
DATASET = new_dataset(shape,parameters,energy)

np.save('file_path/name_file.npy',DATASET)
np.save('file_path/name_file.npy',parameters)



