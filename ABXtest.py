import numpy as np
from random import sample
import tensorflow as tf
file = open('/Users/Hamid/Documents/MachineLearning/NewCIFABX.csv', mode = 'r')
Data_ABX3 = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols = (1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34))
#print(Data_ABX3)
'''
import numpy as np
from random import sample
l = 100 #length of data 
f = 50  #number of elements you need
indices = sample(range(l),f)

train_data = data[indices]
test_data = np.delete(data,indices)

'''

train_data = Data_ABX3[0:138, 0:]
test_data = Data_ABX3[138:,0:]
print(len(Data_ABX3))

learning_rate = 0.01

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 1
batch_size = 100

x = tf.placeholder('float', [None, 27])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([35, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
    	sess.run(tf.global_variables_initializer())

    	for epoch in range(hm_epochs):
    		epoch_loss = 0
    		for _ in range(int(mnist.train.num_examples/batch_size)):
    			epoch_x, epoch_y = train_data.next_batch(batch_size)
    			_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
    			epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:Data_ABX3.test.images, y:Data_ABX3.test.labels}))

train_neural_network(x)