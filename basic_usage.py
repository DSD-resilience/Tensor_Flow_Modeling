# docs/examples/basic_usage.py
import tensorflow as tf

x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])
z = x + y
print(z.numpy())
