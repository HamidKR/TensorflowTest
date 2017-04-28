import tensorflow as tf

filename_queue = tf.train.string_input_producer(["/Users/Hamid/Documents/MachineLearning/NewCIFABX.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [['None'], [0.], [0.], [0.], [0.], [0.], [0.], ['None'], [0.], ['None'], [0.], [0.], [0.], ['None'], [0.], [0.], [0.], ['None'], [0.], [0.], [0.], ['None'], [0.], [0.], [0.], ['None'], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
Name,a,b,c,m_a,m_b,m_c,SpaceGroup,Tabel_Num,el1,p1_x,p1_y,p1_z,el2,p2_x,p2_y,p2_z,el3,p3_x,p3_y,p3_z,el4,p4_x,p4_y,p4_z,el5,p5_x,p5_y,p5_z,Natom,E_g,E_a,OmegaMax,E_b,Density = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([a,b,c,m_a,m_b,m_c,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z,p4_x,p4_y,p4_z,p5_x,p5_y,p5_z,Natom,E_g,OmegaMax,E_b,Density])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(208):
    # Retrieve a single instance:
   example, label = sess.run([features, E_a])

  coord.request_stop()
  coord.join(threads)