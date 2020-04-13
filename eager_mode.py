import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
tfe.enable_eager_execution()

'''
Tensors in Tensorflow

0-d tensor: scalar (number) 
1-d tensor: vector
2-d tensor: matrix
n-d tenso: nd_array
'''

# Constants
a = tf.constant(20, shape=(), dtype=tf.float32, name='a') 
b = tf.constant(15, shape=(), dtype=tf.int8, name='b')
x = tf.divide(a, tf.cast(b, dtype=tf.float32), name='divide_a/b')

for i, j in zip(['a','b','x'],[a,b,x]):
    print(i+' = ', j.numpy())

a = tf.constant(2, shape=(2,1), name='a') 
a_ = tf.constant(2, shape=(1,2), name='a_') 
b = tf.constant([[0, 1], [2, 3]], dtype=tf.float32, name='b') 
x = tf.multiply(tf.cast(a, dtype=tf.float32), b, name='mul_ab')
x_ = tf.multiply(a_, b, name='mul_ab_') # This will give you an error

print(x)

# Random Variables
seed = tf.compat.v1.set_random_seed(10)
help(tf.random_normal)

x = tf.random_normal(shape=[8], mean=20, stddev=5, dtype=tf.float32, seed=seed, name='rand_norm')
print(x)

# Filling with either zeros or ones

input_tensor = tf.ones(shape=(3000,3000), dtype=tf.float32, name=None)
another_input = tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
x = tf.add(input_tensor, another_input) 
y = tf.multiply(input_tensor, another_input)
z = tf.matmul(input_tensor, another_input, name='mat_mult')

bin_mat = [[True, True, True],  [True, True, True],  [True, True, True]]
zeros = tf.zeros_like(bin_mat, dtype=None, name=None, optimize=True)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Variables

s = tf.Variable(2, name="scalar") 
m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
W = tf.Variable(tf.zeros([5,5]))

# create variables with tf.get_variable 
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]])) 
W = tf.get_variable("big_matrix", shape=(5, 5), initializer=tf.zeros_initializer())


tf.compat.v1.disable_eager_execution()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Sessions
s = tf.get_variable("scalar", initializer=tf.constant(100), trainable=True)
t = tf.get_variable("matrix", initializer=tf.constant(2, shape=(2,1), name='a'))
sess1 = tf.Session() 
sess1.run([s.initializer,t.initializer]) 
print(sess1.run(tf.multiply(s,t)))
print(sess1.run(s.assign_add(50)))
print(sess1.run(tf.multiply(s,t)))

init = tf.compat.v1.global_variables_initializer()
sess2 = tf.Session()
sess2.run(init)
print(sess2.run(tf.multiply(s,t))) 

sess1.close()
sess2.close()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Placeholders = Think about them as the formal parameters defined in a function

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = tf.multiply(a, b)
with tf.Session() as sess:
    print(sess.run(c, {a: [1, 2, 3]}))
