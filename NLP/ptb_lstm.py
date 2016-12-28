import numpy as np
import tensorflow as tf
import tensorlayer as tl

import time

flags = tf.flags
flags.DEFINE_string("model", "small","A type of model. Possible options are: small, medium, large.")
FLAGS = flags.FLAGS

def main(_):
	#print('MAIN')
	#print(FLAGS.model)
	if FLAGS.model == 'small' :
		batch_size = 30
		num_steps = 20
	elif FLAGS.model == 'medium' :
		batch_size = 30
		num_steps = 30
	elif FLAGS.model == 'large' :
		batch_size = 30
		num_steps = 45
	else:
		raise ValueError('Model size wrong!')

	train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()
	print('len(train_data) {}'.format(len(train_data))) # len(train_data) 929589
	print('len(valid_data) {}'.format(len(valid_data))) # len(valid_data) 73760
	print('len(test_data)  {}'.format(len(test_data)))  # len(test_data)  82430
	print('vocab_size      {}'.format(vocab_size))      # vocab_size      10000	

	input_feature = tf.placeholder(tf.int32, [batch_size, num_steps])
	input_label = tf.placeholder(tf.int32, [batch_size, num_steps])

	def lstm_model(feature, is_train, num_steps, reuse):
		print("\nis_train : %s, num_steps : %d, reuse : %s" % (num_steps, is_training, reuse) )
		# initializer = tf.random_uniform_initializer(init_scale, init_scale) # no use

	sess = tf.InteractiveSession()

if __name__ == "__main__":
	tf.app.run()