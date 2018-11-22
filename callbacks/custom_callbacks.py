import tensorflow as tf
from keras.callbacks import Callback, TensorBoard, BaseLogger
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import time


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


class AdjustBatchSize(Callback):
    def __init__(self, schedule):
        self.schedule = schedule
        super(AdjustBatchSize, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        if self.schedule(epoch):
            self.model.batch_size = self.schedule(epoch)


class AdjustLearningRate(Callback):
    def __init__(self, schedule):
        self.schedule = schedule
        super(AdjustLearningRate, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.model.optimizer.lr, self.schedule(epoch))


class LogBatchSize(TensorBoard):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.epoch_writer = tf.summary.FileWriter(self.log_dir)
        super(LogBatchSize, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        batch_size = self.model.batch_size
        summary = tf.Summary(value=[tf.Summary.Value(tag="batch_size", simple_value=batch_size)])
        self.epoch_writer.add_summary(summary, epoch)
        self.epoch_writer.flush()


class TrainingMonitor(BaseLogger):
    def __init__(self, hyperparameter, figPath):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.hyperparameter = hyperparameter

    def on_train_begin(self, logs={}):
        self.H = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.end_time = time.time()
        t = self.H.get("duration", [])
        t.append(int(self.end_time - self.start_time))
        self.H["duration"] = t

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        l = self.H.get(self.hyperparameter, []) 
        if self.hyperparameter == "learning_rate":
            l.append(float(K.get_value(self.model.optimizer.lr)))
        elif self.hyperparameter == "batch_size":
            l.append(self.model.batch_size)
        self.H[self.hyperparameter] = l

        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.figPath + ".png")
            plt.close()

            N = np.arange(0, len(self.H[self.hyperparameter]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H[self.hyperparameter], label=self.hyperparameter)
            plt.title("{} [Epoch {}]".format(self.hyperparameter, epoch))
            plt.xlabel("Epoch #")
            plt.ylabel(self.hyperparameter)
            plt.legend()
            plt.savefig(self.figPath + "_{}.png".format(self.hyperparameter))
            plt.close()

            N = np.arange(0, len(self.H["duration"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["duration"], label="duration")
            plt.title("Time taken per epoch [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Duration (secs)")
            plt.legend()
            plt.savefig(self.figPath + "_duration.png")
            plt.close()
