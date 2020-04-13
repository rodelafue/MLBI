#%% [markdown]
## This code shows you how to use the pdbpp libraby
#
# This library can be used for debugging your tensoflow code and the procedure implemented here is known as
#[postmorten debugging](href=https://almarklein.org/pm-debugging.html).
# 
# You can find the library [here](https://docs.python.org/3/library/pdb.html), and for more ***Markdown Styling*** you must vist [this](https://www.markdownguide.org/basic-syntax/) page.

#%% [markdown]
### Importing the required modules
#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#%% [markdown]
# I got this example from the the [tensorflow website](https://www.tensorflow.org/guide/function). 
# Just make sure you check the dbpp page to understand how to use

#* where, 
#* list,
#* up, 
#* down, 
#* next, 
#* step, 
#* continue.

# Now, let's load the function

#%%
@tf.function
def f(x):
  if x > 0:
    # Try setting a breakpoint here!
    # Example:
    import pdb
    pdb.set_trace() # where, list, up, down, next, step, continue
    x = x + 1
    pdb.set_trace()
  return x

#%% [markdown]
#You have to configure the ***experimental_run_functions_eagerly***, to make ensure
#that the traced graph function is deactivated. Tensorflow states the following on its [website](https://www.tensorflow.org/api_docs/python/tf/config/experimental_run_functions_eagerly):

#> Calling tf.config.experimental_run_functions_eagerly(True) will make all invocations of tf.function run eagerly instead of running as a traced graph function.This can be useful for debugging or profiling. For example, let's say you implemented a simple iterative sqrt function, and you want to collect the intermediate values and plot the convergence. Appending the values to a list in @tf.function normally wouldn't work since it will just record the Tensors being traced, not the values. Instead, you can do the following.

#%%
tf.config.experimental_run_functions_eagerly(True)

# You can now set breakpoints and run the code in a debugger.
f(tf.constant(1))

tf.config.experimental_run_functions_eagerly(False)

#%% [markdown]
# Finally, if you want to clear the screen, just run the following
#%%
os.system('cls')

