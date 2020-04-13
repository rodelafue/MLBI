import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # INFO AND WARNING

import tensorflow as tf 
print(tf.__version__)
import tensorflow.contrib.eager as tfe 
tfe.enable_eager_execution()
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

x = tfe.Variable(2.0, dtype = tf.float32, trainable=True) 
y = tf.constant(7, shape=(), dtype=tf.float32, name='y') 

def loss(x, y):
    return (y - x ** 2) ** 2

grad = tfe.implicit_gradients(loss)
print(loss(x=x, y=y))  
print(grad(x=x, y=y))

tf.compat.v1.disable_eager_execution()
print("Eager mode: ", tf.executing_eagerly())

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# The Example from the previous class

# CONSTANTS
a = tf.constant(3, name='a', dtype=tf.float32)

# VARIABLES
x = tf.Variable(5, name='x', dtype=tf.float32) # expected: [tf.float32, tf.float64, tf.float16, tf.bfloat16]
y = tf.Variable(3, name='y', dtype=tf.float32)
z = tf.Variable(2, name='z', dtype=tf.float32)

# OPERATIONS
u = tf.multiply(y, z, name='u_mult')
v = tf.add(u, x, name='v_add')   # f(x,y,z)= a*((y*z)+ x)
j = tf.multiply(v,a,'j_mult') 

# Initializing the variables
init = tf.compat.v1.global_variables_initializer()
opt = tf.train.GradientDescentOptimizer(0.1).minimize(j)

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(30):
        print("x = %.4f, y = %.4f, z = %.4f, loss = %.4f " % (x.eval(), y.eval(), z.eval(), j.eval()))
        sess.run(opt)
