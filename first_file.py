import numpy as np

# a=np.arange(20)
# print(a)

# a,st=np.linspace(1, 10, num=50, endpoint=True, retstep=True, dtype=None)
# print(a,st)
# a=np.linspace(1,10,6)
# print(a)

# l=[1,2,3,4,5]
# b=np.array(l)
# print(l)

#**************************reshape**************************
# d=np.arange(24)
# d=d.reshape(4,6)
# # print(d)
# # # print(d.ndim)
# # print(d.shape)
#
# e=np.arange(24,48)
# e=e.reshape(6,4)
# # print(e)
#
# print(np.matmul(d,e))





# a=np.arange(12)
# c=a.reshape(3,4)
# print(c.ndim)
# print(a.shape)

# b=np.arange(12,24)
# b=b.reshape(3,4)
# print(np.matmul(c,b))

#***************************slicing*********************************
# d=np.arange(24)
# d=d.reshape(4,6)
# print(d)
# print(d[:,1])
# print(d[:,1].reshape(len(d[:,1]),1))

# print(c[:,1].reshape(len(c[:,1]),1))
# print(c)

#****************************matrices****************************

# a=np.ones((3,4))
# print(a)



# a=np.zeros((3,4))
# print(a)


# a=np.identity(4)
# print(a)


# a=np.eye(4, M=None, k=-1, dtype=float, order='C')
# print(a)


# a=np.arange(10)
# b=np.copy(a)
# print(b)

# np.isnan()
# np.unique()

# print(np.may_share_memory(c,b))
