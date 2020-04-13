import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
I was having FutureWarnings from tensorflow, but reading some post online
I found out that it was because of the numpy version. I just did the following
inside of the tensorflow environment

pip uninstall numpy 
pip install numpy==1.16.4

The problem is gone!!!!
'''

import tensorflow as tf 
print(tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# CONSTANTS
a = tf.constant(3, name='a', dtype=tf.int8)

# VARIABLES
x = tf.Variable(5, name='x', dtype=tf.int8)
y = tf.Variable(3, name='y', dtype=tf.int8)
z = tf.Variable(2, name='z', dtype=tf.int8)

# OPERATIONS
u = tf.multiply(y, z, name='u_mult')
v = tf.add(u, x, name='v_add')   # f(x,y,z)= a*((y*z)+ x)
j = tf.multiply(v,a,'j_mult') 

# Initializing the variables
init = tf.compat.v1.global_variables_initializer()

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run(init)
    writer = tf.compat.v1.summary.FileWriter('graphs/basic_ng_graph', sess.graph)
    print(sess.run(j))
    print(u.eval())
    writer.close()

#tensorboard --logdir="graphs/basic_ng_graph"
