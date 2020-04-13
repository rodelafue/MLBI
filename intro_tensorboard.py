import os
import logging
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
print("Version: ", tf.__version__)

# The function to be traced.
@tf.function
def my_persistent_function(x, y):
    a = tf.constant(2, shape=(), dtype=tf.float32, name='a') 
    b = tf.constant(3, shape=(), dtype=tf.float32, name='b')
    c = tf.constant(3, shape=(), dtype=tf.float32, name='c')
    x2 = tf.math.square(x, name='x2')
    y2 = tf.math.square(y, name='y2')
    first = tf.math.multiply(a, x2, name='ax2')
    second = tf.multiply(tf.multiply(b, x, name='bx'),y, name='bxy')
    third = tf.math.multiply(c,y2, name='cy2')
    return  tf.add(tf.add(first,second, name='first_second'),third, name='all_together')

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs\\func\\%s' % stamp
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.Variable(2, dtype=tf.float32, name="x") 
y = tf.Variable(2, dtype=tf.float32, name="y")

# Bracket the function call with
tf.summary.trace_on(graph=True, profiler=True)

# Call only one tf.function when tracing.
z = my_persistent_function(x, y)

with writer.as_default():
    tf.summary.trace_export(
      name = "my_persistent",
      step = 0,
      profiler_outdir = logdir)

#tensorboard --logdir="YOUR\PATH\TO_DIR"
