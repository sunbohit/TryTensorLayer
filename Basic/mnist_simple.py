import numpy as np
import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

feat_train, lab_train, feat_valid, lab_valid, feat_test, lab_test = tl.files.load_mnist_dataset(shape=(-1,784))

input_feature = tf.placeholder(tf.float32, [None,784] , name='input_feature')
input_label = tf.placeholder(tf.int64, [None,], name='input_label')

# print(type(input_feature)) # <class 'tensorflow.python.framework.ops.Tensor'>
# print(input_feature._shape) # (?, 784)

network = tl.layers.InputLayer(inputs = input_feature, name ='input_layer')
#print(network.all_layers) # []
#print(network.all_params) # []
#print(network.all_drop) # {}

#print(type(network)) # <class 'tensorlayer.layers.InputLayer'>

network = tl.layers.DropoutLayer(network, keep=0.8, name='dropout_layer_1')
#print(network.all_layers) # [<tf.Tensor 'dropout_layer_1/mul:0' shape=(?, 784) dtype=float32>]
#print(network.all_params) # []
#print(network.all_drop) # {<tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float32>: 0.8}

network = tl.layers.DenseLayer(network, n_units=800,act = tf.nn.relu, name='dense_relu_1')
#print(network.all_layers) # [<tf.Tensor 'dropout_layer_1/mul:0' shape=(?, 784) dtype=float32>, <tf.Tensor 'dense_relu_1/Relu:0' shape=(?, 800) dtype=float32>]
#print(network.all_params) # [<tensorflow.python.ops.variables.Variable object at 0x7faf0bf06470>, <tensorflow.python.ops.variables.Variable object at 0x7faf0bf06198>]
#print(network.all_drop) # {<tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float32>: 0.8}

network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout_layer_2')

network = tl.layers.DenseLayer(network, n_units=800,act = tf.nn.relu, name='dense_relu_2')

network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout_layer_3')

network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity, name='output_layer')

sess.close()