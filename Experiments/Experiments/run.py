import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([.4], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)

initVars = tf.global_variables_initializer()

session = tf.Session()
session.run(initVars)

linMod = W * x + b

y = tf.placeholder(dtype=tf.float32)
squared = tf.square(linMod - y)

loss = tf.reduce_sum(squared)

print(session.run(loss, {x: [1, 2, 3, 4], y:[5, 6, 7, 8]}))

optimizer = tf.train.GradientDescentOptimizer(.1)
trainer = optimizer.minimize(loss)

for i in range(0,1000):
    session.run(trainer, {x: [1, 2, 3, 4], y:[5, 6, 7, 8]})

print(session.run(loss, {x: [1, 2, 3, 4], y:[5, 6, 7, 8]}))