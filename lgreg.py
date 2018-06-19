import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns',12)

data1=pd.read_csv('train.csv')
data2=pd.read_csv('test.csv')
# print(data.groupby('Sex').sum())
# print(data.groupby('Embarked').sum())

data3=data1.iloc[:,2:]
data4=data2.iloc[:,1:]

data3['From']='train'
data4['From']='test'
# print(data4)
main=pd.concat([data3,data4],ignore_index=True)
dic={}
def null_fun(col):
    size=len(main[main[col].isna()])
    if size:
        dic[col]=size

col_name=list(main.columns)

main['Fare']=main['Fare'].replace(np.nan,np.mean(main['Fare']))
main['Age']=main['Age'].replace(np.nan,np.mean(main['Age']))
main['Embarked']=main['Embarked'].replace(np.nan,str(main['Embarked'].mode())[5])

main['Pclass']=main['Pclass'].apply(lambda x:str(x))


for i in col_name:
    null_fun(i)

# print(dic)

# print(main['Name'].dtype)

mdf=main[main['From']=='train']
mddf=main[main['From']=='test']

mdf.drop('Ticket',inplace=True,axis=1)
mdf.drop('Name',inplace=True,axis=1)
mdf.drop('Cabin',inplace=True,axis=1)

mddf.drop('Ticket',inplace=True,axis=1)
mddf.drop('Name',inplace=True,axis=1)
mddf.drop('Cabin',inplace=True,axis=1)

mdf1=pd.get_dummies(mdf)
mddf1=pd.get_dummies(mddf)

mdf1.drop('Pclass_1',inplace=True,axis=1)
mdf1.drop('Sex_female',inplace=True,axis=1)
mdf1.drop('Embarked_C',inplace=True,axis=1)

mddf1.drop('Pclass_1',inplace=True,axis=1)
mddf1.drop('Sex_female',inplace=True,axis=1)
mddf1.drop('Embarked_C',inplace=True,axis=1)


x_train,x_test,y_train,y_test=train_test_split(mdf1,data1['Survived'],test_size=0.2,random_state=5)
# print(len(x_train))

reg=LogisticRegression()
model=reg.fit(x_train,y_train, sample_weight=None)
y_pred=model.predict(x_test)
y_pred2=model.predict(mddf1)


data3=pd.read_csv('gender_submission.csv')
data3.drop('Survived',inplace=True,axis=1)

data3['Survived']=y_pred2

temp=mdf1.copy()

temp['Survived']=data1['Survived']

# print(temp.corr())

temp['Family']=data1['SibSp']+data1['Parch']

temp.drop('SibSp',inplace=True,axis=1)
temp.drop('Parch',inplace=True,axis=1)


temp['Title']=data1['Name'].apply(lambda x:x.split(',')[1].split()[0])




temp['Title']=temp['Title'].apply(lambda x:np.where(x=='Mr.' or  x=='Mrs.' or  x=='Miss.' or  x=='Master.',x, 'Others'))

# print(temp['Title'].unique())
# data3.to_csv('output.csv')

import statsmodels.api as smpapa
lm=sm.Logit(data1['Survived'],mdf1)
a=lm.fit()
print(a.summary())

'''
rfe=RFE(reg, 12)
lr=rfe.fit(mdf1, data1['Survived'])
print(mdf1.columns.values.tolist())
print(lr.support_)
print(lr.ranking_)
'''


# print(reg.score(x_test,y_test))    #to calculate accuracy
# print(data3)
