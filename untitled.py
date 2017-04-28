import tensorflow as tf
import numpy as np

#file = open('/Users/Hamid/Documents/MachineLearning/NewCIFABX.csv', mode = 'r')
file = '/Users/Hamid/Documents/MachineLearning/NewCIFABX.csv'
Data_ABX3 = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols = (1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34))
filename_queue = tf.train.string_input_producer([Data_ABX3])
type(Data_ABX3)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
Name,a,b,c,m_a,m_b,m_c,SpaceGroup,Tabel_Num,el1,p1_x,p1_y,p1_z,el2,p2_x,p2_y,p2_z,el3,p3_x,p3_y,p3_z,el4,p4_x,p4_y,p4_z,el5,p5_x,p5_y,p5_z,Natom,E_g,E_a,OmegaMax,E_b,Density = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([a,b,c,m_a,m_b,m_c,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z,p4_x,p4_y,p4_z,p5_x,p5_y,p5_z,Natom,E_g,OmegaMax,E_b,Density])
with tf.Session() as sess:
  # Start populating the filename queue.
  ##coord = tf.train.Coordinator()
  ##threads = tf.train.start_queue_runners(coord=coord)
   for i in range(208):# Retrieve a single instance:
    example, label = sess.run([features, E_a])

  ##coord.request_stop()
  ##coord.join(threads)

learning_rate = 0.01

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 1
batch_size = 100

x = tf.placeholder('float', [None, 27])
y = tf.placeholder('float', [None, 1])

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([26, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    #  learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
    	sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(208)):
                epoch_x, epoch_y = [example, label]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:Data_ABX3.test.images, y:Data_ABX3.test.labels}))

train_neural_network(x)