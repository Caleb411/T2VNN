import tensorflow as tf


def quantile_mae(q):
    def quantile_loss(y_true, y_pred):
        error = tf.subtract(y_true, y_pred)
        loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)
        return loss

    return quantile_loss
