import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sample.csv')
df.plot(kind='bar', x='name', y='age')
plt.show()