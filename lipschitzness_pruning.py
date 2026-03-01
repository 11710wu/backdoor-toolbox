import torch
import torch.nn as nn

"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""


def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                # 检查通道数是否匹配
                if idx >= conv.weight.shape[0]:
                    continue
                # w = conv_weight * (bn_weight / bn_std)
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx] / std[idx]).abs()
                # 计算通道的 Lipschitz 常数
                # 使用 .detach() 避免 UserWarning，因为此处无需保留梯度
                channel_lips.append(torch.linalg.svdvals(w.detach().cpu())[0].item())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips > channel_lips.mean() + u * channel_lips.std())[0]

            params[name + '.weight'][index] = params[name + '.weight'].mean()
            params[name + '.bias'][index] = params[name + '.bias'].mean()

        # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)