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
# Moreover, you can have additional call backs that can be used for granular training and\or visual control with TensorBoard.
# One thing to start with is a prespecified schedule for the learning rate.

# Let's use the same dataset as before

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

#%% [markdown]
# Create a logdir and a writer to observe the learning rate change through the epochs

#%%
logdir = "logs\\tf_keras_lr\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.2
  if epoch > 10:
    learning_rate = 0.02
  if epoch > 20:
    learning_rate = 0.01
  if epoch > 50:
    learning_rate = 0.005

  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

#%% [markdown]
### Then, you will have two callbacks that need to be passed to the model fit callbacks
#%%
lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#%% [markdown]
# Next, the processes continues as before: 1) Define a model, 2) Compile it, and 3) Train it.

#%%
linear_layer = tf.keras.layers.Dense(units=1, input_shape=(1,))
# now the model will take as input arrays of shape (*, 1)
# and output arrays of shape (*, 1)
model = keras.models.Sequential([linear_layer])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, lr_callback],
)

# tensorboard --logdir=.\logs\tf_keras_lr\20200416-170424

#%% [markdown]
# From here, you could save it for either further training or frontend deployment.