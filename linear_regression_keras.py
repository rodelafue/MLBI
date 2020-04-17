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

#%% [markdown]
# Let's setup the logdir were the [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) will be stored 

#%%
logdir = "logs\\tf_keras\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

'''
tf.keras.layers.Dense(
    units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    **kwargs
)
'''
linear_layer = tf.keras.layers.Dense(units=1, input_shape=(1,))
# now the model will take as input arrays of shape (*, 1)
# and output arrays of shape (*, 1)
model = keras.models.Sequential([linear_layer])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

#%% [markdown]
# Now you can check out your model to see if it correctly specified

#%%
model.summary()

#%% [markdown]
### Train and predict
# Now we are ready to train our linear regression model and predict the 20% of out-of-sample values

#%%
training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size = train_size,
    verbose = 0, # Suppress chatty output; use Tensorboard instead
    epochs = 100,
    validation_data = (x_test, y_test),
    callbacks=[tensorboard_callback],
)

print("Average test loss: ", np.average(training_history.history['loss']))

print('Prediction: {}'.format(model.predict(x_test)))
predictions = model.predict(x_test)

#%% [markdown]

# Remember you can alwas check the tensorboard callbacks in
# tensorboard --logdir=.logs\tf_keras\20200416-174419

### Save and restore the model
# It is advised to reset your metrics before saving so that loaded model has same state,
# since metric states are not preserved by Model.save_weights. Then, you can save the model using TensorFlow format.
# Finally, you can check that the restored model predict with the same precision as before. Additionally, something important
# to notice is that the optimizer state is preserved as well so you can resume training where you left off.

#%%
model.reset_metrics()

# Export the model to a SavedModel
model.save('saved\\tf_format\\', save_format='tf')

# Recreate the exact same model
new_model = keras.models.load_model('saved\\tf_format\\')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)