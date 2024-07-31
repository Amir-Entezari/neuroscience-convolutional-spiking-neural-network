import torch
import numpy as np


class TimeToFirstSpike(torch.nn.Module):
    def __init__(self, time_window, theta, epsilon=1e-3):
        self.time_window = time_window
        self.theta = theta
        self.epsilon = epsilon

    def __call__(self, data):
        original_shape, original_size = data.shape, data.numel()
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if data.dim() > 1:
            print("Data must be converted to vector first.")
            data = data.view((-1,))

        # self.encoded_spikes = torch.zeros((self.duration,) + self.data.shape, dtype=torch.bool)
        encoded_spikes = torch.zeros((self.time_window,) + data.shape, dtype=torch.bool)

        data = (data - data.min()) / (data.max() - data.min())
        data = (data * (1 - self.epsilon)) + self.epsilon
        tau = -self.time_window / np.log(self.epsilon / self.theta)
        for t in range(self.time_window):
            # threshold = self.theta * np.exp(-(t + 1) / tau)
            threshold = np.exp(-(t + 1) / tau)
            encoded_spikes[t, :] = data >= threshold
            data[data >= threshold] = 0
        return encoded_spikes.view(self.time_window, *original_shape)

    def add_encoder_info(self,
                         ax,
                         text_x=0,
                         text_y=0.05):
        info = {
            "duration": self.duration,
            "theta": self.theta,
            "epsilon": self.epsilon,
        }
        params_info = f"""{self.__class__.__name__} params:\n"""
        for key, value in info.items():
            params_info += f"{key}: {value}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

