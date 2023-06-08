# TBHCNN
Python codes for the Tight-Binding Hamiltonian parametritzation by Neural networks (Machine Learning) in function of the parameters: a (interatomic distance), $\epsilon_0$ (on-site energy) and t (hopping term)
The program has not a normal github stricture, is not programed in modules and has not correlation between the python codes.

# Requirements
TensorFlow 2.12.0
Numpy 1.23.5

# Overview
Explication of the codes:
- data_generator.py: creates the energy corbes without noise for training NN
- data_with_noise.py: creates the energy corbes with noise for training NN and final program
- Train_Autoencoder.py: with a CNN autoencoder structure and the data with noise, train the NN for denoising the data.
- Train_Encoder.py; combination of CNN and DNN structure to extract the a,t,$\epsilon_0$ parameters for the energy corbes witout noise
- final_program.py: Sumation of the Autoencoder and Encoder trainined with a normalitzation program.

This code is a part of my final degree Work, for more information about one can read the work (not yet published)

# Important

To execute correctly, the file_path and the name_file must be added
