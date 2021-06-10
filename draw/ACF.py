import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller

csv_path = '../data/valve.csv'
df = pd.read_csv(csv_path)
data = df['intemp']
print(adfuller(data))
lag_acf = acf(data, nlags=1000)
plt.style.use('ggplot')
plt.figure(figsize=(6, 3))
plt.plot(lag_acf)
plt.xlim([0,1000])
plt.xlabel('Time Lag (half hour)')
plt.ylabel('Autocorrelation')
plt.show()
