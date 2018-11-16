import tensorflow as tf
from keras.callbacks import TensorBoard
from keras import backend as K


class LogLearningRate(TensorBoard):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.batch_writer = tf.summary.FileWriter(self.log_dir)
        self.step = 0
        super(LogLearningRate, self).__init__()

    def on_batch_end(self, batch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        summary = tf.Summary(value=[tf.Summary.Value(tag="learning_rate", simple_value=lr)])
        self.batch_writer.add_summary(summary, self.step)
        self.batch_writer.flush()
        self.step += 1
