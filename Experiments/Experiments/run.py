import tensorflow as tf

c1 = tf.constant(3.0)
c2 = tf.constant(4.0)

session = tf.Session()
print(session.run([c1, c2]))

print(session.run(tf.add(c1, c2)))