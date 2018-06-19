import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# p_s=pd.Series([1,2,3,4],index=['a','b','c','d'])
#
# print(p_s)


# dic={'Country':['USA','England','India','China','Afghanistan'],'Capital':['Washington, D.C.','London','Delhi','Beijing','Kabul'],'Language':['American English','English','Hindi','Mandarin','Urdu']}

# country_info=pd.DataFrame(dic,index=[1,2,3,4,5])
# print(country_info)

# print(country_info['Language'])
# print(country_info.loc[1])      #default 1,2
# print(country_info.iloc[0])     #runs on numerical values 0,1,2
# print(country_info.iloc[3:,0:])  #iloc is used for slicing
# print(country_info.loc[4:,'Country'])

data=pd.read_csv('train.csv')     # to read csv files


# print(data.head())
# print(data.tail())
# print(data.describe())
# print(data.at[10,'Name'])      #same as loc
# print(data.iat[10,3])          #same as iloc (uses index)

# dict1={4:2001,5:3000}

# data.rename(columns={'Name':'ttname'},index=dict1, inplace=True)   #for coloumn
# data.rename(index=dict1, inplace=True)                    #for rows
# print(data)
# print(data.at[10,'ttname'])
# print(data.at[3000,'ttname'])
# print(data['ttname'])

# data.drop(columns='Sex', inplace=True)   #to remove any coloumn, use index to remove row
# print(data.at[3,'Sex'])

# reduction=lambda x:x[0]                    #lambda function
# print(data['Sex'].apply(reduction))        #to apply any function on a given column

# data['family']=data['SibSp']+data['Parch']    #to add two or more coloumn
# print(data.loc[888,['SibSp','Parch','family']])
# print(data['family'])

#drop_duplicates
# def tt(s):
#     for 'Mr' in s:
#         return 'Mr'# def tt(s):
#     for 'Mr' in s:
#         return 'Mr'
#

# reduction=lambda x:x[0]+x[1]+x[2]
# print(data['Name'].apply(reduction))

# def title(row):
#     if row['Sex']=='male':
#         return 'Mr'
#     if row['Sex']=='female':
#         return 'Mrs'



# data['Title']=data['Name'].apply(lambda x:x.split(',')[1].split()[0])
#
# print(data['Title'].unique())



#date=2/6/1018

data1=pd.read_csv('train/user_data.csv')
# print(data1.columns)
#
data2=pd.read_csv('train/problem_data.csv')
# print(data2.columns)
#
data3=pd.read_csv('train/train_submissions.csv')
# print(data3.columns)



#***********************merge*******************************
# data=pd.merge(data1,data3)
# # print(data.columns)
# db=pd.merge(data,data2)
# db.to_csv('temp.csv')
# print(db.columns)


dic=pd.DataFrame({'Country':['USA','England','India','China','Afghanistan'],'Capital':['Washington, D.C.','London','Delhi','Beijing','Kabul'],'Language':['American English','English','Hindi','Mandarin','Urdu']})
dic2=pd.DataFrame({'Country':['bangkok','pakistan','Russia','Singapore','thailand'],'Capital':['Washington, D.C.','modinagar','karachi','istanbul','Kabul'],'Language':['urdu','English','tamil','punjabi','Urdu']})
dic3=pd.DataFrame({'Country':['bangkok','pakistan','Australia','Singapore','thailand'],'Currency':['rupee','dollar','pound','euro','dharm']})
# dic3.set_index('Country', drop=True, append=False, inplace=True, verify_integrity=False)
# dic2.set_index('Country', drop=True, append=False, inplace=True, verify_integrity=False)
# print(pd.concat([dic,dic2],ignore_index=True))



# db2=pd.merge(dic2,dic3,how='right')     #inner,left, right, outer
# print(db2)
# print(dic2.join(dic3),lsuffix='Country')
# print(dic2.join(dic3,lsuffix='_l',rsuffix='_r'))
# print(dic3.join(dic2,on='Country'))


# data['Survived'][data['Survived']==0]='0'
# data['Survived'][data['Survived']==1]='1'
# m_s=len(data[data['Sex']=='male'][data['Survived']=='1'])
# f_s=len(data[data['Sex']=='female'][data['Survived']=='1'])
#
#
# plt.bar(['Male','Female'],[m_s,f_s])
# plt.show()

# plt.hist(data['Pclass'],data['PassengerId'])
# plt.show()


# data['Title']=data['Name'].apply(lambda x:x.split(',')[1].split()[0])
# #
# title_list=data['Title'].unique()
# #
# def save_ls(l):
#     c_l=[]
#     for i in l:
#         c_l.append(len(data[data['Title']==i][data['Survived']==1]))
#     return c_l
#
# count_list=save_ls(title_list)
#
# plt.bar(title_list,count_list)
# plt.show()



data.loc[pd.isna(data['Age']),'Age']=np.mean(data['Age'])
print(data[pd.isna(data['Age'])]['Age'])



sigmoid funnction
