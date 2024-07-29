import torch
import torch.nn.functional as F
import numpy as np


def DoGFilterRGB(sigma1, sigma2, kernel_size, mode, channel_mode):
    dict_mode = {"on-center": 1, "off-center": -1}
    ch_dict = {"RG": [0, 1], "GR": [1, 0], "BY": [0, 1, 2], "YB": [1, 2, 0]}

    channels = len(ch_dict[channel_mode])
    filt = torch.zeros((channels, 1, kernel_size, kernel_size))
    sigma_lst = [sigma1, sigma2, sigma2] if channel_mode != "YB" else [sigma1, sigma1, sigma2]

    for c in range(channels):
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - (kernel_size - 1) / 2
                y = j - (kernel_size - 1) / 2
                sum_x_y = -((x ** 2 + y ** 2) / (2 * sigma_lst[c] ** 2))
                dog = np.exp(sum_x_y)
                filt[c, 0, i, j] = dog
        filt[c, 0] -= filt[c, 0].mean()

    return filt * dict_mode[mode]


def conv2dDoG(img, filt, stride, channel_mode):
    ch_dict = {"RG": [0, 1], "GR": [1, 0], "BY": [0, 1, 2], "YB": [1, 2, 0]}
    pad = (filt.shape[-1] - 1) // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='constant', value=0)
    result = torch.zeros_like(img)

    for idx, c in enumerate(ch_dict[channel_mode]):
        result[:, c, :, :] = F.conv2d(img_padded[:, c:c + 1, :, :], filt[idx:idx + 1], stride=stride)

    return result
