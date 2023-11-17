from torch import Tensor
import numpy as np
from matplotlib import pyplot as plt


def matrix_heatmap(harvest: np.array, save_path='./heatmap_of_matrix.png'):
    plt.imshow(harvest)
    plt.tight_layout()
    plt.colorbar()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def tensor_heatmap(x: Tensor, save_path='./heatmap.png', show=False):
    plt.imshow(x.cpu().numpy())
    plt.tight_layout()
    plt.colorbar()
    plt.axis('off')
    if show:
        plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
