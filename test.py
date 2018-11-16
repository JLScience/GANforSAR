# import h5py
# print(h5py.__version__)

# l = [3, 5, 9]
# print(str(l))

import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('train_output.txt')
print(data.shape)
x = np.arange(0, data.shape[0], 1)
plt.plot(x, data[:, 1])
plt.show()