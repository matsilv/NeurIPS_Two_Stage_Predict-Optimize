import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('kp-capacity-100-mse-history.pkl', 'rb') as file:
    mse = pickle.load(file)

with open('kp-capacity-100-regret-history.pkl', 'rb') as file:
    regret = pickle.load(file)

epochs = len(mse[0])
for trial in mse[1:]:
    assert len(trial) == epochs

for trial in regret:
    assert len(trial) == epochs

epochs = list(range(1, epochs + 1))
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.errorbar(epochs, np.mean(mse, axis=0), yerr=np.std(mse, axis=0), fmt='-o', capsize=5, label='MSE')
plt.title('Mean and Std Dev of MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.errorbar(epochs, np.mean(regret, axis=0), yerr=np.std(regret, axis=0), fmt='-o', capsize=5, label='Regret', color='orange')
plt.title('Mean and Std Dev of Regret over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Regret')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
