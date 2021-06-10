import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tcn import TCN
from tensorflow.keras.models import *
from main.model import T2V
from main.config import get_param
from utils import generator, rse, corr
from utils.deepar.loss import quantile_mae


def getData():
    # 数据读取和预处理
    csv_path = '../data/valve.csv'
    df = pd.read_csv(csv_path)
    data = np.array(df)
    data = np.array(data[:, 2:3], dtype='float')  # 数据删除时间列

    # 对数据进行归一化
    mean = data[:get_param('train_len')].mean(axis=0)
    data -= mean
    std = data[:get_param('train_len')].std(axis=0)
    data /= std

    test_gen = generator(data,
                         lookback=get_param('lookback'),
                         delay=get_param('delay'),
                         min_index=get_param('train_len') + get_param('val_len'),
                         max_index=None,
                         step=get_param('step'),
                         batch_size=get_param('batch_size'))

    # 准备预测数据
    X = []
    y = []

    howmanybatch = (get_param('test_len') - get_param('lookback')) // get_param('batch_size')  # 需要预测多少个batch
    for test_one in test_gen:
        X.append(test_one[0])
        y.append(test_one[1])
        howmanybatch = howmanybatch - 1
        if howmanybatch == 0:
            break

    return np.vstack(X), np.hstack(y), std[0], mean[0]


if __name__ == '__main__':
    # 获取数据和温度的标准差
    test_X, test_y, std, mean = getData()

    test_y = test_y * std + mean
    plt.figure(figsize=(6, 3))
    plt.plot(range(200), test_y[2300:2500], 'black', label='Actual', linewidth=0.75)

    # 误差评估
    test_predicts = []
    output = open("评价指标.txt", "w")
    qs = [0.50]
    for q in qs:
        # 加载模型
        _custom_objects = {
            "T2V": T2V,
            "TCN": TCN,
            "quantile_loss": quantile_mae(q)
        }
        model = load_model("../saved/T2V_NN_%d_%d.h5" % (q*100, get_param('delay')), custom_objects=_custom_objects)
        test_predict = model.predict(test_X) * std + mean
        test_predicts.append(test_predict)

        # 评估
        print("q : %0.2f" % (q), file=output)
        print("quantile_loss : %0.4f" % (quantile_mae(q)(test_y, test_predict.reshape(-1))), file=output)

        if q == 0.5:
            print("rse : %0.4f" % (rse(test_y, test_predict.reshape(-1))), file=output)
            print("corr : %0.4f" % (corr(test_y, test_predict.reshape(-1))), file=output)

        print("***********************************", file=output)

    output.close()

    # 预测结果部分展现
    plt.plot(range(200), test_predicts[0][2300:2500], label='Predict', linewidth=0.75)
    # plt.fill_between(x=range(500),
    #                  y1=test_predicts[0][2300:2800].reshape(-1),
    #                  y2=test_predicts[4][2300:2800].reshape(-1),
    #                  color='khaki',
    #                  alpha=0.5)
    # plt.fill_between(x=range(500),
    #                  y1=test_predicts[1][2300:2800].reshape(-1),
    #                  y2=test_predicts[3][2300:2800].reshape(-1),
    #                  color='salmon',
    #                  alpha=0.5)
    plt.xlabel('Horizons: 200')
    plt.ylabel('Inlet Valve Temperature')
    plt.ylim([31,36.5])
    plt.grid(True)
    # plt.legend()
    plt.show()
