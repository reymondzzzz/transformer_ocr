from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

__all__ = ['plot', 'seed_everything_deterministic', 'get_checkpoint_callback']


def plot(embeds, labels, fig_path='./example.pdf'):
    print('Inside plot, saving path=', fig_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    #    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)


def seed_everything_deterministic(seed):
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_callback(callbacks) -> Optional[ModelCheckpoint]:
    try:
        return next((c for c in callbacks if type(c) == ModelCheckpoint))
    except StopIteration:
        return None


def load_module(module_filename: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(__name__, str(module_filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
