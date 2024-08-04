import matplotlib.pyplot as plt
import numpy as np
import imageio
from IPython.display import Image as dispImage


def add_encoder_raster_plot(
        spikes,
        ax,
        s=5,
        **kwargs):
    x, y = np.where(spikes.reshape(spikes.shape[0], -1) == True)
    ax.scatter(x, y, s=s, **kwargs)
    # Plot the raster plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron ID')
    ax.legend(loc='upper right')


