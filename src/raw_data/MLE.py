import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
np.set_printoptions(suppress=True)
data_length = 150

x = pd.read_csv(f"raw_data_{data_length}.csv", header=None)
y = pd.read_csv(f"raw_label_{data_length}.csv", header=None)
X = x.to_numpy()
Y = y.to_numpy()
X = X.reshape((-1, 9, data_length))

# test
# x = np.delete(x, slice(0, 300), 0)
# y = np.delete(y, slice(0, 300), 0)
sample_0 = Y[:, -1] == 0
sample_1 = Y[:, -1] == 1
x0 = X[sample_0]
x1 = X[sample_1]
y0 = Y[sample_0]
y1 = Y[sample_1]
np.random.seed(0)
new_sample_1 = np.random.choice(x1.shape[0], 700 - y0.shape[0], replace=None)
x1 = x1[new_sample_1]
y1 = y1[new_sample_1]
x = np.vstack((x1, x0))
y = np.vstack((y1, y0))
print(x.shape, y.shape)

# sample = 20
# x = x[sample].reshape((1, 9, 250))
# y = y[sample]

def mle_compute(v, data_length):
    mlist = []
    mle_total = 0

    for i in range(50):
        m = i
        # print(np.linalg.norm(v[:, m + 1] - v[:, m]))
        if (np.linalg.norm(v[:, m + 1] - v[:, m]) > 1e-4) & (np.linalg.norm(v[:, m + 1] - v[:, m]) < 1e-2):
            mlist.append(m)
        else:
            break
    # print('#################################################################')
    if len(mlist) == 0:
        mle_total = 100
        return mle_total

    for k in range(data_length - 100, data_length - 1 - max(mlist)):
        mle = 0
        for m in mlist:
            dif = np.linalg.norm(v[:, k + m] - v[:, k + m - 1]) / (np.linalg.norm(v[:, m + 1] - v[:, m]))
            if dif > 1e-2:
                mle_m = np.log(dif)
                mle += mle_m
        mle /= (k - 1) * 0.01 * len(mlist)
        # print(mle)
        mle_total += mle

    mle_total /= 100 - max(mlist)

    return mle_total


label = np.empty((x.shape[0], 2))
for i in range(x.shape[0]):
    mle = mle_compute(x[i], data_length)
    label[i, 1] = mle
    # if ((mle > 6) | (x[i, :, 220:].min() < 0.1)) & (mle != 100):
    if (mle > 0.8) & (mle != 100):  # 250, 0.8; 150, 0.7
    # if ((mle > 0.7) & (mle != 100)) | (x[i, :, data_length - 30:].min() < 0.1):
        label[i, 0] = 0
    elif ((mle < -0.8) & (mle > -10)) & (x[i, :, data_length - 30:].min() > 0.6):  # 250, -1.0; 150, -1.2
        label[i, 0] = 1
    else:
        label[i, 0] = -1

sample_0 = label[:, 0] == 0
sample_1 = label[:, 0] == 1
sample_a = label[:, 0] == -1
y = np.hstack((label, y))
# print(y)
x1 = x[sample_1]
y1 = y[sample_1]
x0 = x[sample_0]
y0 = y[sample_0]
xa = x[sample_a]
ya = y[sample_a]
print(y1.shape)
print(y0.shape)
print(y1)
print(y0)
# plt.plot(range(250), x0[0].T)
# plt.show()
x = np.vstack((x1, x0))
y = np.vstack((y1, y0))

# random sampling
sample = np.random.choice(X.shape[0], 140, replace=False)
x = X[sample]
y = Y[sample]

x = x.reshape(x.shape[0] * x.shape[1], data_length)
xa = xa.reshape(xa.shape[0] * xa.shape[1], data_length)
np.savetxt(f'data_{data_length}.csv', x, delimiter=",")
np.savetxt(f'label_{data_length}.csv', y, delimiter=",")

