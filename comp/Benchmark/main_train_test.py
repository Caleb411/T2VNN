import numpy as np
import matplotlib.pyplot as plt
from main.config import get_param
from utils.deepar.loss import quantile_mae
from utils import rse, corr, generator, get_data


if __name__ == '__main__':

    data, mean, std = get_data()

    # 数据提取
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

    test_predicts = []
    output_file = open("评价指标.txt", "w")
    qs = [0.01, 0.25, 0.50, 0.75, 0.99]
    for q in qs:
        test_predict = np.vstack(X)[:, -1, 0] * std[0] + mean[0]
        test_predicts.append(test_predict)

        # 评估
        print("q : %0.2f" % (q), file=output_file)
        print("quantile_loss : %0.4f" % (quantile_mae(q)(test_y, test_predict.reshape(-1))), file=output_file)

        if q == 0.5:
            print("rse : %0.4f" % (rse(test_y, test_predict.reshape(-1))), file=output_file)
            print("corr : %0.4f" % (corr(test_y, test_predict.reshape(-1))), file=output_file)

        print("***********************************", file=output_file)

    output_file.close()

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
