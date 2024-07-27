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


class PoissonDecoder(torch.nn.Module):
    """
    Decodes spike trains encoded by a Poisson encoder into original pixel values.

    The decoder estimates the original pixel values by calculating the average rate of spikes
    over a given time window, adjusted by the scaling factor used during encoding.

    Args:
        time_window (int): The interval of the coding used in the encoder.
        ratio (float): The scale factor for the probability of spiking used in the encoder.
    """

    def __init__(self, time_window, ratio):
        super().__init__()
        self.time_window = time_window
        self.ratio = ratio

    def __call__(self, spikes):
        if type(spikes) is tuple:
            return tuple([self(sub_spikes) for sub_spikes in spikes])

        original_shape = spikes.shape[1:]
        spikes_flat = spikes.view(self.time_window, -1)

        # Calculate the average number of spikes over the time window
        spike_counts = spikes_flat.sum(dim=0)

        # Estimate the firing rate (average spikes per time step)
        firing_rate = spike_counts.float() / self.time_window

        # Reverse the scaling applied during encoding
        estimated_values = firing_rate / self.ratio

        # Reshape back to the original input shape
        return estimated_values.view(original_shape)
