import numpy as np


def plot_kernel_3d(ax, kernel, cmap='viridis', ax_label=True, **kwargs):
    # kernel = kernel.cpu().numpy()
    height = kernel.shape[0]
    width = kernel.shape[1]
    x = np.linspace(-height // 2, height // 2, height)
    y = np.linspace(-width // 2, width // 2, width)
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, kernel, cmap=cmap, **kwargs)

    if ax_label:
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
