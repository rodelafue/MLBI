import tensorflow as tf
logdir = "./logs/summary/"
writer = tf.summary.create_file_writer(logdir)

x = tf.Variable(2, dtype=tf.float32, name="x") 
y = tf.Variable(2, dtype=tf.float32, name="y")

@tf.function
def my_persistent_function(x, y):
    a = tf.constant(2, shape=(), dtype=tf.float32, name='a') 
    b = tf.constant(3, shape=(), dtype=tf.float32, name='b')
    c = tf.constant(3, shape=(), dtype=tf.float32, name='c')
    return  a*x**2 + b*x*y + c*y**2

# Bracket the function call with
trainable_variables = [x, y]
opt = tf.optimizers.Adam(learning_rate=0.1)
for step in tf.range(100, dtype=tf.int64):
    with tf.GradientTape() as tape:
        with writer.as_default():
            loss = my_persistent_function(x=x, y=y)
            gradients = tape.gradient(loss, trainable_variables)
            #with writer.as_default():
            if step.numpy() % 10 == 0:
                tf.summary.scalar('loss', loss, step=step.numpy())
                tf.summary.scalar('x', x, step=step.numpy())
                tf.summary.scalar('y', y, step=step.numpy())
        writer.flush()
    opt.apply_gradients(zip(gradients, trainable_variables))

#tensorboard --logdir="YOUR\PATH\TO_DIR"
