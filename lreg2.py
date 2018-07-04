import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_columns',12)

data1=pd.read_csv('train.csv')
data2=pd.read_csv('test.csv')

data1['Family']=data1['SibSp']+data1['Parch']
data2['Family']=data2['SibSp']+data2['Parch']

data1['Title']=data1['Name'].apply(lambda x:x.split(',')[1].split()[0])
data2['Title']=data2['Name'].apply(lambda x:x.split(',')[1].split()[0])

data1['Title']=data1['Title'].apply(lambda x:np.where(x=='Mr.' or  x=='Mrs.' or  x=='Miss.' or  x=='Master.',x, 'Others'))
data2['Title']=data2['Title'].apply(lambda x:np.where(x=='Mr.' or  x=='Mrs.' or  x=='Miss.' or  x=='Master.',x, 'Others'))

data1.drop('Ticket',inplace=True,axis=1)
data1.drop('Name',inplace=True,axis=1)
data1.drop('Cabin',inplace=True,axis=1)
data1.drop('SibSp',inplace=True,axis=1)
data1.drop('Parch',inplace=True,axis=1)

data2.drop('Ticket',inplace=True,axis=1)
data2.drop('Name',inplace=True,axis=1)
data2.drop('Cabin',inplace=True,axis=1)
data2.drop('SibSp',inplace=True,axis=1)
data2.drop('Parch',inplace=True,axis=1)

data3=data1.iloc[:,2:]
data4=data2.iloc[:,1:]



data3['From']='train'
data4['From']='test'

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


mdf=main[main['From']=='train']
mddf=main[main['From']=='test']

mdf1=pd.get_dummies(mdf)
mddf1=pd.get_dummies(mddf)

mdf1.drop('Pclass_1',inplace=True,axis=1)
mdf1.drop('Sex_female',inplace=True,axis=1)
mdf1.drop('Embarked_C',inplace=True,axis=1)
mdf1.drop('Title_Master.',inplace=True,axis=1)
mdf1.drop('From_train',inplace=True,axis=1)
mdf1.drop('Embarked_Q',inplace=True,axis=1)   #after watching in summary
mdf1.drop('Embarked_S',inplace=True,axis=1)   #after watching in summary
mdf1.drop('Title_Others',inplace=True,axis=1)   #after watching in summary

mddf1.drop('Pclass_1',inplace=True,axis=1)
mddf1.drop('Sex_female',inplace=True,axis=1)
mddf1.drop('Embarked_C',inplace=True,axis=1)
mddf1.drop('Title_Master.',inplace=True,axis=1)
mddf1.drop('From_test',inplace=True,axis=1)
mddf1.drop('Embarked_Q',inplace=True,axis=1)   #after watching in summary
mddf1.drop('Embarked_S',inplace=True,axis=1)   #after watching in summary
mddf1.drop('Title_Others',inplace=True,axis=1)   #after watching in summary


x_train,x_test,y_train,y_test=train_test_split(mdf1,data1['Survived'],test_size=0.2,random_state=5)
reg=KNeighborsClassifier(n_neighbors=5,weights='distance')
model=reg.fit(x_train,y_train)
# y_pred=model.predict(x_test)

y_pred=model.predict(mddf1)

data_csv=pd.read_csv('gender_submission.csv')
data_csv.drop('Survived',inplace=True,axis=1)
data_csv['Survived']=y_pred
data_csv.to_csv('output3.csv')
print(data_csv)

# print(mddf1.columns)
# print(data4.columns)

print(reg.score(x_test,y_test))

# import statsmodels.api as sm
# lm=sm.Logit(data1['Survived'],mdf1)
# a=lm.fit()
# print(a.summary())

#hello
