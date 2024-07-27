import torch


class Latency2Intensity(torch.nn.Module):
    """
    Latency to intensity decoding.
    Earlier spikes correspond to stronger intensity values.

    Args:
        time_window (int): The interval of the coding.
        min_val (float): Minimum possible value of the output image. Default is 0.0.
        max_val (float): Maximum possible value of the output image. Default is 1.0.
    """

    def __init__(self, time_window, min_val=0.0, max_val=1.0):
        super().__init__()
        self.time_window = time_window
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, spikes):
        # Finding the first spike in each pixel
        first_spikes, _ = (spikes == 1).max(dim=0)
        spike_times, _ = (spikes * torch.arange(1, self.time_window + 1, device=spikes.device).view(-1, 1, 1)).max(
            dim=0)

        # Adjusting for pixels that never spike
        spike_times = torch.where(first_spikes, spike_times, torch.tensor(self.time_window, device=spikes.device))

        # Normalize spike times to intensity values
        img = (self.time_window - spike_times.float()) / self.time_window
        img = img * (self.max_val - self.min_val) + self.min_val

        return img

