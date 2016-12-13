# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.stdout = sys.stderr
import tensorflow as tf
import numpy as np
#from tensorflow.models.rnn.ptb import reader
import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool('debug', False, 'More debug info in Tensorboard')
flags.DEFINE_string('cost_function', 'default', 'Which cost function to use')
flags.DEFINE_string('optimizer', 'GradientDescentOptimizer', 'Which optimizer to use')
flags.DEFINE_bool('non_rnn_in_fp32', True, 'Perform non-rnn layers in fp32')
flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None, "Model output directory.")
flags.DEFINE_float("reg_term", 0.0, "L2 regularization of parameters")
flags.DEFINE_float("init_scale", 0.0, "initialization for weights will be [-init_scale, init_scale]")
flags.DEFINE_float("initial_lr", 0.0, "learning rate for 0 epoch")
flags.DEFINE_integer("max_valid_increases", 1, "max number of times validation error can go up before action is taken")
flags.DEFINE_integer("gseed", 1, "graph level random seed")

FLAGS = flags.FLAGS
print('Model is: %s' % FLAGS.model)
print('use_fp16 is: %s' % FLAGS.use_fp16)
print('cost function is: %s' % FLAGS.cost_function)
print('optimizer is: %s' % FLAGS.optimizer)
print('do non rnn layers in fp32: %s' % FLAGS.non_rnn_in_fp32)
print('output debug info: %s' % FLAGS.debug)
print('l2 regularizer weight: %s' % FLAGS.reg_term)
print('weight initializer init_scale: %s' % (FLAGS.init_scale if FLAGS.init_scale else 'will use model default'))
print('initial learning rate: %s' % FLAGS.initial_lr)
print('Gseed: %s' % FLAGS.gseed)


def data_type(is_lstm_layer=False):
    if not is_lstm_layer and FLAGS.non_rnn_in_fp32:
        return tf.float32
    else:
        return tf.float16 if FLAGS.use_fp16 else tf.float32

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor.
        This is also quite expensive.
    """
    with tf.name_scope('summaries'):
        s_var = tf.cast(var, tf.float32)
        amean = tf.reduce_mean(tf.abs(s_var))
        tf.summary.scalar('amean/' + name, amean)
        mean = tf.reduce_mean(s_var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(s_var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(s_var))
        tf.summary.scalar('min/' + name, tf.reduce_min(s_var))
        tf.summary.histogram(name, var)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, debug=False):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.size = size = config.hidden_size
    vocab_size = config.vocab_size
    self.num_layers = config.num_layers

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type(is_lstm_layer=False))
    inputs = tf.nn.embedding_lookup(embedding, self._input_data, name="inputs_to_rnn")
    if debug:
        variable_summaries(inputs, "inputs_to_rnn")

    if is_training and config.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, config.keep_prob)

    rnn = tf.contrib.cudnn_rnn.CudnnLSTM(config.num_layers, size, size, input_mode='linear_input', direction='unidirectional',
                                         dropout=config.keep_prob, seed=0, seed2=0, ttype=data_type(is_lstm_layer=True))
    params_size_t = rnn.params_size()
    self._initial_input_h = tf.placeholder(data_type(is_lstm_layer=True), shape=[config.num_layers, batch_size, size]) #self._initial_input_h = tf.Variable(tf.zeros([config.num_layers, batch_size, size]))
    self._initial_input_c = tf.placeholder(data_type(is_lstm_layer=True), shape=[config.num_layers, batch_size, size]) #self._initial_input_c = tf.Variable(tf.zeros([config.num_layers, batch_size, size]))
    #self.params = tf.get_variable("params", [params_size_t], validate_shape=False, dtype=data_type(is_lstm_layer=False))
    self.params = tf.Variable(tf.random_uniform([params_size_t], minval=-config.init_scale, maxval=config.init_scale, dtype=data_type(is_lstm_layer=True)), validate_shape=False)
    self.params_size_t = rnn.params_size()

    outputs, output_h, output_c = rnn(is_training=is_training, input_data=tf.transpose(tf.cast(inputs, dtype=data_type(is_lstm_layer=True)), [1, 0, 2]), input_h=self.input_h,
                                     input_c=self.input_c, params=self.params)

    self._output_h = output_h
    self._output_c = output_c

    output = tf.reshape(tf.concat(1, tf.transpose(outputs, [1, 0, 2])), [-1, size])

    if debug:
        variable_summaries(output, 'multiRNN_output')

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type(is_lstm_layer=False))
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type(is_lstm_layer=False))
    logits = tf.matmul(output if output.dtype == data_type(is_lstm_layer=False) else tf.cast(output, data_type(is_lstm_layer=False)), softmax_w) + softmax_b

    if debug:
       variable_summaries(logits, 'logits')

    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type(is_lstm_layer=False))])

    self._cost = cost = tf.reduce_sum(loss) / batch_size
    if FLAGS.cost_function == 'avg':
      self._cost_to_optimize = cost_to_optimize = tf.reduce_mean(loss)
    else:
      self._cost_to_optimize = cost_to_optimize = cost

    tvars = tf.trainable_variables()
    for v in tvars:
        cost_to_optimize += FLAGS.reg_term * tf.cast(tf.nn.l2_loss(v), dtype=data_type(False)) / (batch_size*config.num_steps)
        self._cost_to_optimize = cost_to_optimize

    if debug:
        tf.summary.scalar('cost no regularization', cost)
        tf.summary.scalar('cost_to_optimize', cost_to_optimize)

    #self._final_state = state

    if not is_training:
        self.merged = tf.summary.merge_all()
        return

    self._lr = tf.Variable(0.0, trainable=False, dtype=data_type(is_lstm_layer=False))
    #if debug:
    #        tf.scalar_summary('learning rate', self._lr)

    #tvars = tf.trainable_variables()
    type2vars = dict()
    print("**************************")
    print("Trainable Variables")
    print("**************************")
    for var in tvars:
        print('Variable name: %s. With dtype: %s and shape: %s' % (var.name, var.dtype, var.get_shape()))
        if var.dtype not in type2vars:
            type2vars[var.dtype] = [var]
        else:
            type2vars[var.dtype].append(var)

    print("**************************")
    print("Gradients Variables")
    print("**************************")
    _grads = tf.gradients(cost_to_optimize, tvars)
    type2grads = dict()
    for g in _grads:
        print('Gradient name: %s. With dtype: %s' % (g.name, g.dtype))
        if g.dtype not in type2grads:
            type2grads[g.dtype] = [g]
        else:
            type2grads[g.dtype].append(g)

    type2clippedGrads = dict()
    for dtype in type2grads:
        cgrads, _ = tf.clip_by_global_norm(type2grads[dtype], config.max_grad_norm)
        type2clippedGrads[dtype] = cgrads


    if debug:
        for (gkey, vkey) in zip(type2clippedGrads.keys(),type2vars.keys()):
            for (clipped_gradient, variable) in zip(type2clippedGrads[gkey], type2vars[vkey]):
                variable_summaries(clipped_gradient, "clipped_dcost/d"+variable.name)
                variable_summaries(variable, variable.name)


    if FLAGS.optimizer == 'MomentumOptimizer':
        optimizer = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.9)
    elif FLAGS.optimizer == 'AdamOptimizer':
        optimizer = tf.train.AdamOptimizer()
    elif FLAGS.optimizer == 'RMSPropOptimizer':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self._lr)
    elif FLAGS.optimizer == 'AdagradOptimizer':
        optimizer = tf.train.AdagradOptimizer(learning_rate=self._lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

    allgrads = []
    allvars = []
    for dtype in type2clippedGrads:
        allgrads += type2clippedGrads[dtype]

    #WARNING: key order assumption
    for dtype in type2vars:
        allvars += type2vars[dtype]

    self._train_op = optimizer.apply_gradients(zip(allgrads, allvars))

    self._new_lr = tf.placeholder(dtype=data_type(False), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self.merged = tf.summary.merge_all()

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def input_h(self):
    return self._initial_input_h

  @property
  def input_c(self):
    return self._initial_input_c

  @property
  def output_h(self):
      return self._output_h

  @property
  def output_c(self):
      return self._output_c

  @property
  def input(self):
    return self._input

  @property
  def cost(self):
    return self._cost

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
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
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    #init_scale = 0.04
    init_scale = 0.004
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
    vocab_size = 10000


class XLargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 3
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class XXLargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 4
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, data, eval_op=None, verbose=False, epoch_ind=0):
    """Runs the model on the given data."""
    start_time = time.time()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    costs = 0.0
    iters = 0
    if eval_op is not None:
      fetches = [model.cost, model.output_h, model.output_c, eval_op]
    else:
      fetches = [model.cost, model.output_h, model.output_c]

    h = np.zeros(shape=(model.num_layers, model.batch_size, model.size), dtype=np.float32)
    c = np.zeros(shape=(model.num_layers, model.batch_size, model.size), dtype=np.float32)

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.input_c] = c
        feed_dict[model.input_h] = h

        if eval_op is not None:
          cost, h, c, _ = session.run(fetches, feed_dict)
        else:
          cost, h, c = session.run(fetches, feed_dict)

        costs += cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    print("TOTAL EPOCH TIME: %.3f seconds" % (time.time() - start_time))
    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "xlarge":
        return XLargeConfig()
    elif FLAGS.model == "xxlarge":
        return XXLargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.gseed)
        if FLAGS.init_scale != 0.0:
            initializer = tf.random_uniform_initializer(-1*FLAGS.init_scale,
                                                        FLAGS.init_scale)
        else:
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, debug=FLAGS.debug)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config)

        #with tf.name_scope("Test"):
        #    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #        mtest = PTBModel(is_training=False, config=eval_config)

        per_epoch_train_loss_update = tf.placeholder(tf.float32, shape=[])
        per_epoch_train_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name='Epoch_train_loss', validate_shape=False)
        tf.summary.scalar("Training Perplexity", per_epoch_train_loss)
        per_epoch_train_loss_update_op = tf.assign(per_epoch_train_loss, per_epoch_train_loss_update)

        per_epoch_valid_loss_update = tf.placeholder(tf.float32, shape=[])
        per_epoch_valid_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name='Epoch_train_loss', validate_shape=False)
        tf.summary.scalar("Validation Perplexity", per_epoch_valid_loss)
        per_epoch_valid_loss_update_op = tf.assign(per_epoch_valid_loss, per_epoch_valid_loss_update)
        #

        summary = tf.summary.merge_all()

        prev_validation_error = float("inf")
        validation_err_went_up_counter = 0
        saver = tf.train.Saver()
        #summary_writer = tf.train.SummaryWriter(logdir=FLAGS.save_path, graph=tf.get_default_graph())
        sv = tf.train.Supervisor(logdir=FLAGS.save_path, is_chief=True,
                                 save_model_secs=0, saver=saver, save_summaries_secs=0) #
        if FLAGS.initial_lr != 0.0:  # we'll do 0 epoch
            erange = [-1] + range(config.max_max_epoch)
        else:
            erange = range(config.max_max_epoch)
        path_to_latest_checkpoint = ""

        with sv.managed_session() as session:
            for i in erange:
                if i != -1:
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)
                else: #very first epoch
                    m.assign_lr(session, FLAGS.initial_lr)

                print("Epoch: %d Learning rate: %.8f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, eval_op=m.train_op, verbose=True, epoch_ind=i)

                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, valid_data)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                if valid_perplexity < prev_validation_error:
                    prev_validation_error = valid_perplexity
                    validation_err_went_up_counter = 0
                    path_to_latest_checkpoint = sv.saver.save(sess=session, save_path=FLAGS.save_path+"/model",
                                                              global_step=i)
                    print("Saved currently best model to: %s" % path_to_latest_checkpoint)
                else:
                    validation_err_went_up_counter += 1
                    if validation_err_went_up_counter > FLAGS.max_valid_increases:
                        print("EARLY STOPPING!!! Restoring from %s" % (path_to_latest_checkpoint))
                        sv.saver.restore(session, path_to_latest_checkpoint)

                session.run(per_epoch_valid_loss_update_op, feed_dict={per_epoch_valid_loss_update: valid_perplexity})
                session.run(per_epoch_train_loss_update_op, feed_dict={per_epoch_train_loss_update: train_perplexity})
                #sv.summary_writer.add_summary(session.run(summary), i)

            #test_perplexity = run_epoch(session, mtest, test_data)
            #print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    tf.app.run()
    