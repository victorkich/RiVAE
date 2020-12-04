from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('output.csv', index_col=False)
x = np.arange(0, df.shape[0])
plt.plot(x, df['train_loss'], label='train_loss')
plt.plot(x, df['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

