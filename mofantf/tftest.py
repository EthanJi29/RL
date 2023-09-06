import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

x_data = np.random.rand(100).astype(float)
y_data = x_data*0.1+0.3


Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = x_data*Weights + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step, sess.run(Weights), sess.run(biases))

