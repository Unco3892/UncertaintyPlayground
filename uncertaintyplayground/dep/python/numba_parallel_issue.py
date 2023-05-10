import numpy as np
import pickle
import os.path
import numba as nb
from numba import njit, prange

# read necessary variables coming from another script
def read_pickle(var_name):
  if not isinstance(var_name, str):
      raise ValueError("Variable name must be a string")
  
  file_name = f"{var_name}.pkl"
  
  if not os.path.isfile(file_name):
      raise ValueError(f"File {file_name} does not exist")
  
  with open(file_name, 'rb') as f:
      var_value = pickle.load(f)
  
  globals()[var_name] = var_value

read_pickle('train_smp_size')
read_pickle('custom_sd')
read_pickle('R')
read_pickle('z_t')
read_pickle('n_z_outcomes')

# normal function which works
def sample_z_r(n_hiddens, z_means, a_size=train_smp_size, sampling_R=R, desired_sd=custom_sd):
  # this is require for R -> Python as it doesn't always turn the objects to integers
  n_hiddens = int(n_hiddens)
  a_size = int(a_size)
  sampling_R = int(sampling_R)
  
  # define the `z_R` array
  desired_z_R = np.empty((a_size, n_hiddens, sampling_R))
  for l in range(n_hiddens):
      for r in range(sampling_R):
          desired_z_R[:, l, r] = np.random.normal(loc=z_means[:, l], scale=desired_sd, size=a_size)
          ## desired_z_R is N x L x sampling_R
  return desired_z_R

print(n_z_outcomes[1])
print(z_t[1])
print(sample_z_r(n_z_outcomes[1], z_means=z_t[1])) # take the first instance to try, however, in the real context with apply a loop like

#-------------------------------------------------------------------------------
# now with parallelism it doesn't work
@nb.njit(parallel = True)
def sample_z_r(n_hiddens, z_means, a_size=train_smp_size, sampling_R=R, desired_sd=custom_sd):
  # convert n_hiddens, a_size, and sampling_R to 16-bit integers
  n_hiddens = np.int16(n_hiddens) #can also be `int()`
  a_size = np.int16(a_size)
  sampling_R = np.int16(sampling_R)

  # define the `z_R` array
  desired_z_R = np.empty((a_size, n_hiddens, sampling_R))
  
  for l in range(n_hiddens): #after debuggining, substitute this with nb.prange
    for r in range(sampling_R):
      desired_z_R[:, l, r] = np.random.normal(
        loc=z_means[:, l],
        scale=desired_sd,
        size=a_size
        ) ## desired_z_R is N x L x sampling_R
        
  return desired_z_R

# problematic part
print(sample_z_r(n_z_outcomes[1], z_means=z_t[1]))
