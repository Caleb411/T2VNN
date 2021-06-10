import pandas as pd
import matplotlib.pyplot as plt
from tcn import tcn_full_summary
from main.model import T2V_NN
from main.config import get_param, get_params
from utils import generator, get_data


# 绘制学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


if __name__ == '__main__':
    data, mean, std = get_data()

    # 数据提取
    train_gen = generator(data,
                          lookback=get_param('lookback'),
                          delay=get_param('delay'),
                          min_index=0,
                          max_index=get_param('train_len'),
                          shuffle=True,
                          step=get_param('step'),
                          batch_size=get_param('batch_size'))
    val_gen = generator(data,
                        lookback=get_param('lookback'),
                        delay=get_param('delay'),
                        min_index=get_param('train_len'),
                        max_index=get_param('train_len') + get_param('val_len'),
                        step=get_param('step'),
                        batch_size=get_param('batch_size'))

    train_steps = (get_param('train_len') - get_param('lookback')) // get_param('batch_size')
    val_steps = (get_param('val_len') - get_param('lookback')) // get_param('batch_size')

    qs = [0.50]
    for q in qs:
        model = T2V_NN(get_params(), q)

        tcn_full_summary(model, expand_residual_blocks=False)

        history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=get_param('epochs'),
                            validation_data=val_gen, validation_steps=val_steps)
        plot_learning_curves(history)

        # 保存模型
        model.save('../saved/T2V_NN_%d_%d.h5' % (q*100, get_param('delay')))
        print('export model saved.')
