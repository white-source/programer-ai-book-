import tensorflow as tf

class WideAndDeepModel:
    def __init__(self, wide_length, deep_length, deep_last_layer_len, softmax_label):
        self.input_wide_part = tf.placeholder(tf.float32, shape=[None, wide_length], name='input_wide_part')
        self.input_deep_part = tf.placeholder(tf.float32, shape=[None, deep_length], name='input_deep_part')
        self.input_y = tf.placeholder(tf.float32, shape=[None, softmax_label], name='input_y')

        with tf.name_scope('deep_part'):
            w_x1 = tf.Variable(tf.random_normal([wide_length, 64], stddev=0.03), name='w_x1')
            b_x1 = tf.Variable(tf.random_normal([64]), name='b_x1')

            w_x2 = tf.Variable(tf.random_normal([64, deep_last_layer_len], stddev=0.03), name='w_x2')
            b_x2 = tf.Variable(tf.random_normal([deep_last_layer_len]), name='b_x2')

            z1 = tf.add(tf.matmul(self.input_wide_part, w_x1), b_x1)
            a1 = tf.nn.relu(z1)
            self.deep_logits = tf.add(tf.matmul(a1, w_x2), b_x2)

        with tf.name_scope('wide_part'):
            weights = tf.Variable(tf.truncated_normal([deep_last_layer_len + wide_length, softmax_label]))
            biases = tf.Variable(tf.zeros([softmax_label]))

            self.wide_and_deep = tf.concat([self.deep_logits, self.input_wide_part], axis = 1)

            self.wide_and_deep_logits = tf.add(tf.matmul(self.wide_and_deep, weights), biases)
            self.predictions = tf.argmax(self.wide_and_deep_logits, 1, name= "prediction")


        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.wide_and_deep_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")