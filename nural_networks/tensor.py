import tensorflow as tf
# print(tf.Variable(tf.matmul(tf.Variable(tf.random_normal([10,12])),tf.Variable(tf.random_normal([12,10])))))
a=tf.truncated_normal([5,5,1,32],stddev=0.1)
print(a)
