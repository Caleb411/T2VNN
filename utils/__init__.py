import numpy as np
import pandas as pd

from main.config import get_param


def get_data():
    # 数据读取和预处理
    csv_path = 'D:\\PycharmProjects\\T2VNN\\data\\valve.csv'
    df = pd.read_csv(csv_path)
    data = np.array(df)
    data = np.array(data[:, 2:3], dtype='float')  # 数据删除时间列

    # 对数据进行归一化
    mean = data[:get_param('train_len')].mean(axis=0)
    data -= mean
    std = data[:get_param('train_len')].std(axis=0)
    data /= std

    return data, mean, std


# 数据生成器函数
def generator(data, lookback, delay, min_index, max_index, step, shuffle=False, batch_size=128):
    if max_index is None:
        max_index = len(data) - delay
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay - 1][0]
        yield samples, targets, [None]


def rse(y_true, y_pred):
    #
    # The formula is:
    #           K.sqrt(K.sum(K.square(y_true - y_pred)))
    #    RSE = -----------------------------------------------
    #           K.sqrt(K.sum(K.square(y_true_mean - y_true)))
    #
    #           K.sqrt(K.sum(K.square(y_true - y_pred))/(N-1))
    #        = ----------------------------------------------------
    #           K.sqrt(K.sum(K.square(y_true_mean - y_true)/(N-1)))
    #
    #
    #           K.sqrt(K.mean(K.square(y_true - y_pred)))
    #        = ------------------------------------------
    #           K.std(y_true)
    #
    num = np.sqrt(np.mean(np.square(y_true - y_pred), axis=None))
    den = np.std(y_true, axis=None)

    return num / den


def corr(y_true, y_pred):
    #
    # This function calculates the correlation between the true and the predicted outputs
    #
    num1 = y_true - np.mean(y_true, axis=0)
    num2 = y_pred - np.mean(y_pred, axis=0)

    num = np.mean(num1 * num2, axis=0)
    den = np.std(y_true, axis=0) * np.std(y_pred, axis=0)

    return np.mean(num / den)
