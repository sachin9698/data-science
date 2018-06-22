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

# print(mdf1)
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





temp=mdf.copy()
temp2=mddf.copy()


# temp['Survived']=data1['Survived']
# print(temp.corr())

temp['Family']=data1['SibSp']+data1['Parch']
temp2['Family']=data2['SibSp']+temp2['Parch']

print(temp2['Family'])

temp.drop('SibSp',inplace=True,axis=1)
temp.drop('Parch',inplace=True,axis=1)

temp2.drop('SibSp',inplace=True,axis=1)
temp2.drop('Parch',inplace=True,axis=1)

# print(temp2['Family'])
# print(temp2.columns)

# data10=pd.read_csv('test.csv')

# print(data10)

temp['Title']=data1['Name'].apply(lambda x:x.split(',')[1].split()[0])
temp2['Title']=data2['Name'].apply(lambda x:x.split(',')[1].split()[0])

# print(temp2['Title'])


temp['Title']=temp['Title'].apply(lambda x:np.where(x=='Mr.' or  x=='Mrs.' or  x=='Miss.' or  x=='Master.',x, 'Others'))
temp2['Title']=temp2['Title'].apply(lambda x:np.where(x=='Mr.' or  x=='Mrs.' or  x=='Miss.' or  x=='Master.',x, 'Others'))

# print(temp2)

temp1=pd.get_dummies(temp)
# temp3=pd.get_dummies(temp2)

# temp1.drop('Title_Master.',inplace=True,axis=1)
# temp1.drop('Survived',inplace=True,axis=1)

# temp3.drop('Title_Master.',inplace=True,axis=1)
# temp3.drop('Survived',inplace=True,axis=1)

# print(temp1.columns)
# print(temp3.columns)
# print(temp1)



# x_train1,x_test1,y_train1,y_test1=train_test_split(temp1,data1['Survived'],test_size=0.2,random_state=5)
# model1=reg.fit(x_train1,y_train1, sample_weight=None)
# y_pred3=model1.predict(mddf1)

# print(reg.score(x_test1,y_test1))








# print(temp['Title'].unique())


# data3.to_csv('output.csv')



# import statsmodels.api as sm
# lm=sm.Logit(data1['Survived'],temp1)
# a=lm.fit()
# print(a.summary())


'''
rfe=RFE(reg, 12)
lr=rfe.fit(mdf1, data1['Survived'])
print(mdf1.columns.values.tolist())
print(lr.support_)
print(lr.ranking_)
'''


# print(reg.score(x_test,y_test))    #to calculate accuracy
# print(data3)




# data_csv=pd.read_csv('gender_submission.csv')
# data_csv.drop('Survived',inplace=True,axis=1)
# data_csv['Survived']=y_pred3
# data_csv.to_csv('output2.csv')
# print(data_csv)
