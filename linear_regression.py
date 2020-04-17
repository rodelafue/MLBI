#%% [markdown]

## Import the required modules

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("TensorFlow version: ", tf.__version__)

#%% [markdown]
## Create the simplest dataset

# We will have a 1000x2 dataset containing x and y. Then, we are going to split the rows in 80% for training and 20% for testing
#%%
data_size = 1000
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it.
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the output data.
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size, ))

# Split into test and train pairs.
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

class LinearModel(object):    
    def __init__(self):
        self.Weight = tf.Variable(11.0, dtype=tf.float64)
        self.Bias = tf.Variable(12.0, dtype=tf.float64)
        self.trainable_variables = [self.Weight, self.Bias]
    
    def train(self, x, y, lr=0.12, epochs=100, verbose=False):
        # Set up logging.
        y = tf.cast(y,dtype=tf.float64)
        x = tf.cast(x,dtype=tf.float64)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs\\custom_linear_regression\\%s' % stamp
        writer = tf.summary.create_file_writer(logdir)
        @tf.function
        def loss(y,predicted):
            return tf.reduce_mean(tf.square(y - predicted))
        opt = tf.optimizers.Adam(learning_rate=lr)
        with writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)
            for step in tf.range(epochs, dtype=tf.int64):
                with tf.GradientTape() as tape:
                    predicted = self.Bias + self.Weight*x
                    loss_val = loss(y, predicted)
                    gradients = tape.gradient(loss_val, self.trainable_variables)
                    if step.numpy() % 10 == 0:
                        tf.summary.scalar('loss', loss_val, step=step.numpy())
                        tf.summary.scalar('W1', self.Weight, step=step.numpy())
                        tf.summary.scalar('W0', self.Bias, step=step.numpy())
                if verbose:
                    print("Weight = %.4f, Bias = %.4f, loss = %.4f " % (self.Weight.numpy(), 
                            self.Bias.numpy(), loss_val.numpy()))
                opt.apply_gradients(zip(gradients, self.trainable_variables))
            tf.summary.trace_export(name = "my_persistent", step = step, profiler_outdir = logdir)
        writer.close()
        
    def predict(self, x):
        x = tf.cast(x, dtype=tf.float64)
        return self.Bias + self.Weight*x
   
my_linear = LinearModel()
my_linear.train(x_train, y_train)
p=my_linear.predict(x_test)

# tensorboard --logdir=.\logs\custom_linear_regression\20200416-165540