import numpy as np


#a=np.arange(20)
#print(a)

#np.random.seed(34)
# b=np.random.rand(5)
# b=np.random.rand(10,2,2) #(10=no of random no. ,2,2 = size of matrix)
# print(b)

# b=np.random.randn(5,2,2)
# b=np.random.randint(2,10,8)  #(2,10 = range low and high , 8= no. of random no)
# b=np.random.normal(10,3,11)
# print(b)

# np.random.seed(30)
# dt=np.dtype(('string'))
# a=np.random.randint(20,40,20)
# a=np.array(a,dtype=dt)
# print(a)

# a=a.reshape(4,5)
# print(a)
# b=np.arange(10,20)
# print(a.sum(1))  # 0 for coloumn and 1 for row
# print(a[1,2])
# print(a.max())
# print(a.min())
# print(a.argmax()) #return the index of max value
# print(a.argmin()) #return the index of min value

# for i in range(0,20):
#     if(a[i]>=25 and a[i]<=30):
#         print(a[i])

# print(a[(a>=25) & (a<=30)])
# bl=[0,1,3,7]
# print(np.count_nonzero(a[a==25]))


# s=input()
# ms=''
# for i in s:
#     if i.lower() not in 'aeiou':
#         ms+=i
# print(ms)

# a=[1,2,3,4]
# print(a)

#date = 26 may

# a=np.arange(10)
# print(a)
# b=a[1::2]
# c=a[::-1]
# print(c)

# a=np.arange(12)
#
# a=a.reshape(3,4)
# # print(a[::,::-1])    #reverse the elements of row
# # print(a,a[1:list(a.shape)[0]-1,1:list(a.shape)[1]-1])
#
#
# b=np.arange(12,24)

# print(np.column_stack((a,b)))







# dt=np.dtype([('City','S20'),('Population',np.int),('Area',np.int),('State','S20')])
#
#
# data=np.array([('ghaziabad',320,230,'UP'),('gwalior',160,120,'MP'),('chandigarh',65,145,'CH'),('dhanbad',100,147,'JH'),('meerut',148,452,'UP')],dtype=dt)
# print(data['City'])
# np.savetxt('test.csv', data,fmt='%.20s %.3d %.3d %.20s')
# data1=np.loadtxt('test.csv', dtype=dt)
# print(data1)



dt=np.dtype([('City',np.unicode,'S40'),('Population',np.int),('Area',np.int),('State',np.unicode,'S20')])
# #
# #
data=np.array([('ghaziabad',320,230,'UP'),('gwalior',160,120,'MP'),('chandigarh',65,145,'CH'),('dhanbad',100,147,'JH'),('meerut',148,452,'UP')],dtype=dt)
# # print(data['City'])
np.savetxt('test1.csv', data,fmt='%.20s %.3d %.3d %.20s',delimiter=',')
# #data1=np.loadtxt('test1.csv', dtype=None)
#
# #print(data1)
#
# import pandas as pd
