import numpy as np
import matplotlib.pyplot as plt
import sys, gvar

config_np = sys.argv[1]

# Load data
config = np.load(config_np)
L = config.shape[-1]
config = np.exp(1j*config) # convert phases to fields
config = config.astype(np.complex128)
 
wl = config[:,1,:,:]*\
     np.roll(config, shift=-1, axis=-1)[:,0,:,:]*\
     np.roll(config, shift=-1, axis=-2).conj()[:,1,:,:]*\
     config.conj()[:,0,:,:]

wl = np.mean(wl, axis=(-1,-2))
# Perform blocking
block_list = [i for i in range(1, config.shape[0]+1) if config.shape[0]%i == 0] 
yr = [] 
yi = []
for iblock in block_list:
    nconf = wl.shape[0]
    rs = wl.reshape(nconf//iblock, iblock)
    rs = np.mean(rs, axis=-1)
    rs = rs.flatten()
    yr.append(gvar.dataset.avg_data(rs.real))
    yi.append(gvar.dataset.avg_data(rs.imag))
    print(f"\tblock {iblock} = {yr[-1]} + i({yi[-1]}) (num meas. = {len(rs)})")
