import scipy.io
import numpy as np

data = scipy.io.loadmat("simulation_low.mat")

print(data.keys())
Nt = int(np.asarray(data["kgrid_Nt"]).squeeze())
print(Nt)