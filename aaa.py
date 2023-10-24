import numpy as np
import matplotlib.pyplot as plt


data = np.load('diff_xvfi.npy')

plt.scatter(data[:,0], data[:,1])
plt.xlabel("mse_loss")
plt.ylabel("psnr")
plt.show()
print(data.shape)