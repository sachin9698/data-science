import pandas as pd


p_s=pd.Series([1,2,3,4],index=['a','b','c','d'])

dic={'Country':['USA','England','India','China','Afghanistan'],'Capital':['Washington, D.C.','London','Delhi','Beijing','Kabul'],'Language':['American English','English','Hindi','Mandarin','Urdu']}

country_info=pd.DataFrame(dic,index=[1,2,3,4,5])

# print(country_info['Language'])
# print(country_info.loc[2])   #default 0,1,2
# print(country_info.iloc[2])   #runs on numerical values
# print(country_info.iloc[3:,0])  #iloc is used for slicing
# print(country_info.loc[4:,'Country'])

data=pd.read_csv('train.csv')     # to read csv files
# print(data.head())
# print(data.tail())
# print(data.describe())
# print(data.at[10,'Name'])     #same as loc
# print(data.iat[10,3])          #same as iloc

# dict1={4:2001,5:3000}

# data.rename(columns={'Name':'ttname'},index=dict1, inplace=True)

# print(data.at[10,'ttname'])
# print(data.at[3000,'ttname'])

# data.drop(columns='Sex', inplace=True)   #to remove any coloumn, use index to remove row
# print(data.at[3,'Sex'])

# reduction=lambda x:x[0]           #lambda function
# print(data['Sex'].apply(reduction))        #to apply any function on a given column

data['family']=data['SibSp']+data['Parch']    #to add two or more coloumn
print(data.loc[888,['SibSp','Parch','family']])
# print(data['family'])

#drop_duplicates
