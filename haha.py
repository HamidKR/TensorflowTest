import tensorflow as tf
import numpy as np

file = open('/Users/Hamid/Documents/MachineLearning/NewCIFABX.csv', mode = 'r')
Data_ABX3 = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols = (1, 2, 3, 4, 5))

filename_queue = tf.train.string_input_producer([Data_ABX3])


reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
# print tf.shape(col1)

features = tf.concat(0, [col1, col2, col3, col4])
with tf.Session() as sess:
  # Start populating the filename queue.
 # coord = tf.train.Coordinator()
  #threads = tf.train.start_queue_runners(coord=coord)

  for i in range(208):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])

 # coord.request_stop()
  #coord.join(threads)