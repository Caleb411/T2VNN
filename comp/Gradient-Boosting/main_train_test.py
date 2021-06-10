import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from main.config import get_param
from utils import rse, corr, generator, get_data
from utils.deepar.loss import quantile_mae


# 生成测试集
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=True):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


if __name__ == '__main__':

    data, mean, std = get_data()

    X_train, y_train = multivariate_data(data[:, 0], data[:, 0], 0,
                                         get_param('train_len'),
                                         get_param('lookback'),
                                         get_param('delay') - 1,
                                         get_param('step'))

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

    howmanybatch = (get_param('test_len') - get_param('lookback')) // get_param('batch_size')
    for test_one in test_gen:
        X.append(test_one[0])
        y.append(test_one[1])
        howmanybatch = howmanybatch - 1
        if howmanybatch == 0:
            break

    test_y = np.hstack(y) * std[0] + mean[0]
    plt.figure(figsize=(9, 3))
    plt.plot(range(1000), test_y[2000:3000], 'black', label='Actual')

    # 误差评估
    test_predicts = []
    output = open("评价指标.txt", "w")
    qs = [0.01, 0.25, 0.50, 0.75, 0.99]
    for q in qs:
        ridge = GradientBoostingRegressor(loss="quantile", alpha=q)
        ridge.fit(X_train, y_train)
        test_predict = ridge.predict(np.vstack(X)[:, :, 0]) * std[0] + mean[0]
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
    plt.fill_between(x=range(1000),
                     y1=test_predicts[0][2000:3000].reshape(-1),
                     y2=test_predicts[4][2000:3000].reshape(-1),
                     alpha=0.5)
    plt.fill_between(x=range(1000),
                     y1=test_predicts[1][2000:3000].reshape(-1),
                     y2=test_predicts[3][2000:3000].reshape(-1),
                     alpha=0.5)
    plt.xlabel('Horizons: 1000')
    plt.ylabel('Inlet Valve Temperature')
    plt.grid(True)
    # plt.legend()
    plt.show()
