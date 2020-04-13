import os
import logging
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
from tensorflow.python.training import gradient_descent
print("Version: ", tf.__version__)

# The function to be traced.

@tf.function
def my_persistent_function(x, y):
    a = tf.constant(2, shape=(), dtype=tf.float32, name='a') 
    b = tf.constant(3, shape=(), dtype=tf.float32, name='b')
    c = tf.constant(3, shape=(), dtype=tf.float32, name='c')
    return  a*x**2 + b*x*y + c*y**2
    
# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs\\altogether\\%s' % stamp
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.Variable(2, dtype=tf.float32, name="x") 
y = tf.Variable(2, dtype=tf.float32, name="y")

# Bracket the function call with

# Call only one tf.function when tracing.
#z = my_persistent_function(x, y)

# Bracket the function call with
trainable_variables = [x, y]
opt = tf.optimizers.Adam(learning_rate=0.1)
with writer.as_default():
    tf.summary.trace_on(graph=True, profiler=True)
    for step in tf.range(100, dtype=tf.int64):
        with tf.GradientTape() as tape:
            loss = my_persistent_function(x=x, y=y)
            gradients = tape.gradient(loss, trainable_variables)
            if step.numpy() % 10 == 0:
                tf.summary.scalar('loss', loss, step=step.numpy())
                tf.summary.scalar('x', x, step=step.numpy())
                tf.summary.scalar('y', y, step=step.numpy())
        print("x = %.4f, y = %.4f, loss = %.4f " % (x.numpy(), y.numpy(), loss.numpy()))
        opt.apply_gradients(zip(gradients, trainable_variables))
    tf.summary.trace_export(name = "my_persistent", step = step, profiler_outdir = logdir)
writer.close()

#tensorboard --logdir="YOUR\PATH\TO_DIR"

