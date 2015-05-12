import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_PATH1 = '1.txt'
LOG_PATH2 = '2.txt'

logs = [np.loadtxt(LOG_PATH1), np.loadtxt(LOG_PATH2)]

ind = len(logs[0]) - np.argmax(np.diff(logs[0], axis=0)[:, 0][::-1]<0) - 1
ind = ind if ind + 1 != len(logs[0]) else 0


losses = [log[ind:, :] for log in logs]

for ii, model_loss in enumerate(losses):
    epoch_n = model_loss[:, 0]
    val_loss = model_loss[:, 1]
    train_loss = model_loss[:, 2]
    plt.plot(epoch_n, val_loss, label='val loss '+str(ii), lw=2)
    plt.plot(epoch_n, train_loss, label='train loss'+str(ii), lw=2)
    plt.legend()
    plt.ylim([0.42, 0.52])
plt.show()


