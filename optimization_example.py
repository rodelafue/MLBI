import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
import numpy as np
print("Version: ", tf.__version__)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Option 1
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Constants
a = tf.constant(2, shape=(), dtype=tf.float32, name='a') 
b = tf.constant(8, shape=(), dtype=tf.float32, name='a') 

# Variables
x = tf.Variable(3, name='x', trainable=True, dtype=tf.float32)
y = tf.Variable(2, name='y', trainable=True, dtype=tf.float32)

trainable_variables = [x, y]

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
loss = lambda: x**2 + y**2 + a*x + b*y

for k in tf.range(100, dtype=tf.int64):
    print("iter= %s, x = %.4f, y = %.4f, loss = %.4f " % (k.numpy(), x.numpy(), y.numpy(), loss().numpy()))
    opt.minimize(loss, var_list=trainable_variables)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Alternativa 2
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

from tensorflow.python.training import gradient_descent

@tf.function
def f_():
    f = x**2 + y**2 + a*x + b*y
    return f

for _ in tf.range(20, dtype=tf.int64):
    print("x = %.4f, y = %.4f, loss = %.4f " % (x.numpy(), y.numpy(), f_().numpy()))
    opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(f_)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Alternativa 3
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def f_xy(x,y):
    f = x**2 + y**2 + a*x + b*y
    return f

x = tf.Variable(3, name='x', trainable=False, dtype=tf.float32)
y = tf.Variable(2, name='y', trainable=True, dtype=tf.float32)

trainable_variables = [x, y]
opt = tf.optimizers.Adam(learning_rate=0.1)
for step in tf.range(200, dtype=tf.int64):
    with tf.GradientTape() as tape:
        f = f_xy(x=x, y=y)
        gradients = tape.gradient(f, trainable_variables)
    print("x = %.4f, y = %.4f, loss = %.4f " % (x.numpy(), y.numpy(), f.numpy()))
    opt.apply_gradients(zip(gradients, trainable_variables))

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Watching Gradients
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
x = tf.Variable(3, name='x', trainable=True, dtype=tf.float32)
y = tf.Variable(2, name='y', trainable=True, dtype=tf.float32)

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    g.watch(y)
    f = x**2 + y**2 + a*x + b*y
derivatives = g.gradient(f, [x, y])


