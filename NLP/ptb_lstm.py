import numpy as np
import tensorflow as tf
import tensorlayer as tl

import time

flags = tf.flags
flags.DEFINE_string("model", "small","A type of model. Possible options are: small, medium, large.")
FLAGS = flags.FLAGS

def main(_):
	print('MAIN')
	print(FLAGS.model)

if __name__ == "__main__":
	tf.app.run()