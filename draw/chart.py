import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = '../data/valve.csv'
df = pd.read_csv(csv_path)

data = np.array(df)
data = np.array(data[:, 2:], dtype='float')

plt.figure(figsize=(6, 3))
plt.style.use('seaborn-whitegrid')
plt.plot(range(len(data[:, 0])), data[:, 0], label='ground_truth')
plt.ylim([15,45])
plt.xlabel('Time(half hour)')
plt.ylabel('valve')
plt.show()