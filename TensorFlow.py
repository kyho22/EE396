import TensorFlow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#hdl = hidden layers, nodes per layer: 500
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
#Go through batches of 100 to feed in neuronetwork
batch_size = 100

#x = input data (784pixels wide flattend from 28x28)
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
    # input_data * weight + biases

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal(n_classes))}

    #perform calc
    l1 = td.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']) 
    #goes thru activation func
    l1 = tf.nn.relu(l1)

    l2 = td.add(tf.matmul(data, hidden_2_layer['weights']) + hidden_2_layer['biases']) 
    l2 = tf.nn.relu(l2)           

    l3 = td.add(tf.matmul(data, hidden_3_layer['weights']) + hidden_3_layer['biases']) 
    l3 = tf.nn.relu(l3)

    output = td.add(tf.matmul(data, output_layer['weights']) + output_layer['biases']) 

    return output