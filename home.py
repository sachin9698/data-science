import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',12)

data1=pd.read_csv('home/train.csv')
data2=pd.read_csv('home/test.csv')

ms=data1.groupby('MSZoning')
ms=ms.sum()
ms['SalePrice'].plot()
plt.show()
