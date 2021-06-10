params = {
    'train_len': 17434,     # 训练集长度
    'val_len': 5811,        # 验证集长度
    'test_len': 5811,       # 测试集长度

    'lookback': 48,         # 采样长度
    'step': 1,              # 采样间隔
    'delay': 3,             # 延迟步数 horizon>=1 [3,6,9,12]
    'batch_size': 128,      # 批量大小

    'unit': 256,            # 神经网络超参数
    't2v_dim': 64,          # 嵌入层超参数
    'hw': 3,                # AR超参数 [3,6,12,24]

    'epochs': 5             # 训练的轮数
}


def get_param(param):
    return params[param]

def get_params():
    return params
