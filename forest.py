# linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

np.random.seed(100)
c1=np.random.randint(0,100,50)
c2=np.random.randint(1000,5000,50)
dic={'col1':c1,'col2':c2}
data1=pd.DataFrame(dic)
# data2
# print(data1)

data_train,data_test=train_test_split(data1,test_size=0.2,random_state=5)
reg=RandomForestRegressor(1000)
model=reg.fit(np.array(data_train['col1']).reshape(-1,1),data_train['col2'])

y_pred=model.predict(np.array(data_test['col1']).reshape(-1,1))

# print(data_test['col2'])
# print(y_pred)
print(reg.score(np.array(data_test['col1']).reshape(-1,1),data_test['col2']))
plt.scatter(data_test['col1'],data_test['col2'])
# plt.show()
plt.scatter(data_test['col1'],y_pred,color='red')
plt.show()
