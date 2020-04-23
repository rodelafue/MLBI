#Declaring necessary modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
A simple numpy implementation of a XOR gate to understand the backpropagation
algorithm
"""
# Declare the inputs
x = tf.constant([[1,1],[1,0],[0,1],[0,0]], dtype=tf.float64, name='xy')
y = tf.constant([[0],[1],[1],[0]], dtype=tf.float64, name='xor')

#%% [markdown]
### Forward Pass
#%%
class Model(object):
    def __init__(self):
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        samples, features = (4,2) #x.shape
        hidden1_nodes = 2
        hidden2_nodes = 1
        self.theta1 = tf.Variable(tf.random.normal([features+1,hidden1_nodes], dtype= tf.float64) ,name = "Theta1")
        self.theta2 = tf.Variable(tf.random.normal([hidden1_nodes+1,hidden2_nodes], dtype= tf.float64), name = "Theta2")

    def __call__(self, x):
        bias1 = tf.constant([[1],[1],[1],[1]], dtype=tf.float64, name='bias1')
        bias2 = tf.constant([[1],[1],[1],[1]], dtype=tf.float64, name='bias2')
        a0 = tf.concat([bias1,x],1, name='a0')
        z1 = tf.matmul(a0,self.theta1, name='z1')
        a1 = tf.concat([bias2,tf.sigmoid(z1)],1,name='a1')
        z2 = tf.matmul(a1,self.theta2, name='z2')
        a2 = tf.sigmoid(z2, name='a2')
        return a2

def loss(target_y, predicted_y):
    return -tf.reduce_sum(target_y*tf.math.log(predicted_y)+(1-target_y)*tf.math.log(1-predicted_y), axis = 0, name='Cost_function')
    
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
    dThe1, dThe2 = t.gradient(current_loss, [model.theta1, model.theta2])
    model.theta1.assign_sub(learning_rate * dThe1)
    model.theta2.assign_sub(learning_rate * dThe2)
    
def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.2
    if epoch > 3000:
        learning_rate = 0.15
    if epoch > 4000:
        learning_rate = 0.1
    if epoch > 4500:
        learning_rate = 0.05
    return learning_rate
#%% [markdown]
### Optimize
    
#%%
model = Model()
theta1_hist, theta2_hist = [], []
for epoch in range(5000):
    current_loss = loss(y, model(x))
    learning_rate = lr_schedule(epoch)
    train(model, x, y, learning_rate=learning_rate)
    if epoch % 100 == 0:
        theta1_hist.append(model.theta1.numpy())
        theta2_hist.append(model.theta2.numpy())
        print('Epoch %2d: learning_rate=%2.5f, loss=%2.5f' % (epoch, learning_rate, current_loss))


