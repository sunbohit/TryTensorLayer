import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time

flags = tf.flags
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
FLAGS = flags.FLAGS


def main(_):

    if FLAGS.model == "small":
        init_scale = 0.1
        learning_rate = 1.0
        max_grad_norm = 5
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
    elif FLAGS.model == "medium":
        init_scale = 0.05
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 35
        hidden_size = 650
        max_epoch = 6
        max_max_epoch = 39
        keep_prob = 0.5
        lr_decay = 0.8
        batch_size = 20
    elif FLAGS.model == "large":
        init_scale = 0.04
        learning_rate = 1.0
        max_grad_norm = 10
        num_layers = 2
        num_steps = 35
        hidden_size = 1500
        max_epoch = 14
        max_max_epoch = 55
        keep_prob = 0.35
        lr_decay = 1 / 1.15
        batch_size = 20
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

    train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()

    print('len(train_data) {}'.format(len(train_data))) # len(train_data) 929589
    print('len(valid_data) {}'.format(len(valid_data))) # len(valid_data) 73760
    print('len(test_data)  {}'.format(len(test_data)))  # len(test_data)  82430
    print('vocab_size      {}'.format(vocab_size))      # vocab_size      10000

    sess = tf.InteractiveSession()

    # One int represents one word, the meaning of batch_size here is not the
    # same with MNIST example, it is the number of concurrent processes for
    # computational reasons.

    # Training and Validing
    input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    # Testing (Evaluation)
    input_data_test = tf.placeholder(tf.int32, [1, 1])
    targets_test = tf.placeholder(tf.int32, [1, 1])

    def inference(x, is_training, num_steps, reuse=None):
        """If reuse is True, the inferences use the existing parameters,
        then different inferences share the same parameters.

        Note :
        - For DynamicRNNLayer, you can set dropout and the number of RNN layer internally.
        """
        print("\nnum_steps : %d, is_training : %s, reuse : %s" %
                                                (num_steps, is_training, reuse))
        initializer = tf.random_uniform_initializer(init_scale, init_scale)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.EmbeddingInputlayer(
                        inputs = x,
                        vocabulary_size = vocab_size,
                        embedding_size = hidden_size,
                        E_init = tf.random_uniform_initializer(-init_scale, init_scale),
                        name ='embedding_layer')
            
            if is_training:
                network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop1')
            
            network = tl.layers.RNNLayer(network,
                        cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                        cell_init_args={'forget_bias': 0.0},# 'state_is_tuple': True},
                        n_hidden=hidden_size,
                        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                        n_steps=num_steps,
                        return_last=False,
                        name='basic_lstm_layer1')
            lstm1 = network
            
            if is_training:
                network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop2')
            
            network = tl.layers.RNNLayer(network,
                        cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                        cell_init_args={'forget_bias': 0.0}, # 'state_is_tuple': True},
                        n_hidden=hidden_size,
                        initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                        n_steps=num_steps,
                        return_last=False,
                        return_seq_2d=True,
                        name='basic_lstm_layer2')
            lstm2 = network
            
            # Alternatively, if return_seq_2d=False, in the above RNN layer,
            # you can reshape the outputs as follow:
            # network = tl.layers.ReshapeLayer(network,
            #       shape=[-1, int(network.outputs._shape[-1])], name='reshape')
            if is_training:
                network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop3')
            
            network = tl.layers.DenseLayer(network,
                        n_units=vocab_size,
                        W_init=tf.random_uniform_initializer(-init_scale, init_scale),
                        b_init=tf.random_uniform_initializer(-init_scale, init_scale),
                        act = tf.identity, name='output_layer')
        return network, lstm1, lstm2

    # Inference for Training
    network, lstm1, lstm2 = inference(input_data,
                            is_training=True, num_steps=num_steps, reuse=None)
    # Inference for Validating
    network_val, lstm1_val, lstm2_val = inference(input_data,
                            is_training=False, num_steps=num_steps, reuse=True)
    # Inference for Testing (Evaluation)
    network_test, lstm1_test, lstm2_test = inference(input_data_test,
                            is_training=False, num_steps=1, reuse=True)

    sess.run(tf.initialize_all_variables())

    def loss_fn(outputs, targets, batch_size, num_steps):
        # Returns the cost function of Cross-entropy of two sequences, implement
        # softmax internally.
        # outputs : 2D tensor [batch_size*num_steps, n_units of output layer]
        # targets : 2D tensor [batch_size, num_steps], need to be reshaped.
        # batch_size : RNN batch_size, number of concurrent processes.
        # n_examples = batch_size * num_steps
        # so
        # cost is the averaged cost of each mini-batch (concurrent process).
        print()
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [outputs],
            [tf.reshape(targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        cost = tf.reduce_sum(loss) / batch_size
        return cost

    # Cost for Training
    cost = loss_fn(network.outputs, targets, batch_size, num_steps)
    # Cost for Validating
    cost_val = loss_fn(network_val.outputs, targets, batch_size, num_steps)
    # Cost for Testing (Evaluation)
    cost_test = loss_fn(network_test.outputs, targets_test, 1, 1)

    # Truncated Backpropagation for training
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    sess.run(tf.initialize_all_variables())

    network.print_params()
    network.print_layers()
    tl.layers.print_all_variables()

    print("\nStart learning a language model by using PTB dataset")
    for i in range(max_max_epoch):
        # decreases the initial learning rate after several
        # epoachs (defined by ``max_epoch``), by multipling a ``lr_decay``.
        new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        sess.run(tf.assign(lr, learning_rate * new_lr_decay))

        # Training
        print("Epoch: %d/%d Learning rate: %.3f" % (i + 1, max_max_epoch, sess.run(lr)))
        epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
        start_time = time.time()
        costs = 0.0; iters = 0
        # reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data,
                                                    batch_size, num_steps)):
            feed_dict = {input_data: x, targets: y,
                        lstm1.initial_state: state1,
                        lstm2.initial_state: state2,
                        }
            # For training, enable dropout
            feed_dict.update( network.all_drop )
            _cost, state1, state2, _ = sess.run([cost,
                                            lstm1.final_state,
                                            lstm2.final_state,
                                            train_op],
                                            feed_dict=feed_dict
                                            )
            costs += _cost; iters += num_steps

            if step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                    iters * batch_size / (time.time() - start_time)))
        train_perplexity = np.exp(costs / iters)
        print("Epoch: %d/%d Train Perplexity: %.3f" % (i + 1, max_max_epoch,
                                                            train_perplexity))

        # Validing
        start_time = time.time()
        costs = 0.0; iters = 0
        # reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1_val.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2_val.initial_state)
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(valid_data,
                                                    batch_size, num_steps)):
            feed_dict = {input_data: x, targets: y,
                        lstm1_val.initial_state: state1,
                        lstm2_val.initial_state: state2,
                        }
            _cost, state1, state2, _ = sess.run([cost_val,
                                            lstm1_val.final_state,
                                            lstm2_val.final_state,
                                            tf.no_op()],
                                            feed_dict=feed_dict
                                            )
            costs += _cost; iters += num_steps
        valid_perplexity = np.exp(costs / iters)
        print("Epoch: %d/%d Valid Perplexity: %.3f" % (i + 1, max_max_epoch,
                                                            valid_perplexity))


    print("Evaluation")
    # Testing
    # go through the test set step by step, it will take a while.
    start_time = time.time()
    costs = 0.0; iters = 0
    # reset all states at the begining
    state1 = tl.layers.initialize_rnn_state(lstm1_test.initial_state)
    state2 = tl.layers.initialize_rnn_state(lstm2_test.initial_state)
    for step, (x, y) in enumerate(tl.iterate.ptb_iterator(test_data,
                                            batch_size=1, num_steps=1)):
        feed_dict = {input_data_test: x, targets_test: y,
                    lstm1_test.initial_state: state1,
                    lstm2_test.initial_state: state2,
                    }
        _cost, state1, state2 = sess.run([cost_test,
                                        lstm1_test.final_state,
                                        lstm2_test.final_state],
                                        feed_dict=feed_dict
                                        )
        costs += _cost; iters += 1
    test_perplexity = np.exp(costs / iters)
    print("Test Perplexity: %.3f took %.2fs" % (test_perplexity, time.time() - start_time))


if __name__ == "__main__":
    tf.app.run()