import tensorflow as tf
x1 = tf.constant([5,7])
x2 = tf.constant([6,77,9])

print(x1*x2) #prints the tensor and not the result

sess = tf.Session()
output = sess.run(x1+x2)  
print output #actual value
#or print(sess.run(x1*x2)
sess.close()
