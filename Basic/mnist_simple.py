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
#print(type(network)) # <class 'tensorlayer.layers.DropoutLayer'>

network = tl.layers.DenseLayer(network, n_units=800,act = tf.nn.relu, name='dense_relu_2')
#print(type(network)) # <class 'tensorlayer.layers.DenseLayer'>

network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout_layer_3')
#print(type(network)) # <class 'tensorlayer.layers.DropoutLayer'>

network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity, name='output_layer')
#print(type(network)) # <class 'tensorlayer.layers.DenseLayer'>

predict_label = network.outputs
#print(type(predict_label)) # <class 'tensorflow.python.framework.ops.Tensor'>

#print(predict_label._shape) # (?, 10)
#print(input_label._shape) # (?,)
cost = tl.cost.cross_entropy(predict_label, input_label) 
#print(type(cost)) # <class 'tensorflow.python.framework.ops.Tensor'>

train_params = network.all_params
#print(train_params)#[<tensorflow.python.ops.variables.Variable object at 0x7f98d9396080>, <tensorflow.python.ops.variables.Variable object at 0x7f98d9396208>, <tensorflow.python.ops.variables.Variable object at 0x7f98d9396780>, <tensorflow.python.ops.variables.Variable object at 0x7f98d8336c50>, <tensorflow.python.ops.variables.Variable object at 0x7f98d9376518>, <tensorflow.python.ops.variables.Variable object at 0x7f98d83559b0>]
train_op = tf.train.AdamOptimizer(0.0001).minimize(cost, var_list=train_params)

correct = tf.equal(tf.argmax(predict_label, 1), input_label)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess.run(tf.initialize_all_variables())

network.print_params()
network.print_layers()



sess.close()