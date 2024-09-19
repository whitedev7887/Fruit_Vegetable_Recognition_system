import tensorflow as tf

hello=tf.tf.constant("hello world")

with tf.compat.v1.Session()as sess:
    result=sess.run(hello)
    print(result.decode)