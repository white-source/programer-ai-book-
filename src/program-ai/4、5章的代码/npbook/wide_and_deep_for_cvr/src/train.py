import tensorflow as tf
import data_helpers
import os
import datetime
import time
from WideandDeepModel import WideAndDeepModel

# Data loading params
tf.flags.DEFINE_string("train_dir", "../data/cvr_train_data.csv", "Path of train data")
tf.flags.DEFINE_integer("wide_length", 261, "Path of train data")
tf.flags.DEFINE_integer("deep_length", 261, "Path of train data")
tf.flags.DEFINE_integer("deep_last_layer_len", 32, "Path of train data")
tf.flags.DEFINE_integer("softmax_label", 1, "Path of train data")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs")
tf.flags.DEFINE_integer("display_every", 50, "Number of iterations to display training info.")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with.")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps")



# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train():
    with tf.device('/cpu:0'):
        x, y = data_helpers.load_data_and_labels(FLAGS.train_dir)

    print('-' * 120)
    print(x.shape)
    print(y.shape)
    print('-' * 120)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            model = WideAndDeepModel(
                wide_length=FLAGS.wide_length,
                deep_length=FLAGS.deep_length,
                deep_last_layer_len=FLAGS.deep_last_layer_len,
                softmax_label=FLAGS.softmax_label
            )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            #
            # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_dir = '/Users/asukapan/workspace/all_codes/iscp_all_codes/src/wide_and_deep_for_cvr/src/model/'
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                feed_dict = {
                    model.input_wide_part: x_batch,
                    model.input_deep_part: x_batch,
                    model.input_y: y_batch
                }

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, model.loss, model.accuracy], feed_dict)

                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, auc {:G}".format(time_str, step, loss, accuracy))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

        save_path = saver.save(sess, checkpoint_prefix)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()