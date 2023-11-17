import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .ColorUtils import get_datetime_str

modes = ['3D', 'contour', '2D']


class Landscape4Model():
    def __init__(self, model: nn.Module, loss: callable, mode='2D',
                 save_path='./landscape/'):
        '''

        :param model:
        :param loss: given a model, return loss
        '''
        self.save_path = save_path
        assert mode in modes
        self.mode = mode
        self.model = model
        self.loss = loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def synthesize_coordinates(self, x_min=-0.02, x_max=0.02, x_interval=1e-3,
                               y_min=-0.02, y_max=0.02, y_interval=1e-3):
        x = np.arange(x_min, x_max, x_interval)
        y = np.arange(y_min, y_max, y_interval)
        self.mesh_x, self.mesh_y = np.meshgrid(x, y)

    @torch.no_grad()
    def draw(self):
        self._find_direction()
        z = self._compute_for_draw()
        self.draw_figure(self.mesh_x, self.mesh_y, z)

    @torch.no_grad()
    def _find_direction(self):
        self.x_unit_vector = {}
        self.y_unit_vector = {}
        for name, param in self.model.named_parameters():
            self.x_unit_vector[name] = torch.randn_like(param.data)
            self.y_unit_vector[name] = torch.randn_like(param.data)

    @torch.no_grad()
    def _compute_for_draw(self):
        result = []
        if self.mode == '2D':
            self.mesh_x = self.mesh_x[0]
            for i in tqdm(range(self.mesh_x.shape[0])):
                now_x = self.mesh_x[i]
                loss = self._compute_loss_for_one_coordinate(now_x, 0)
                result.append(loss)
        else:
            for i in tqdm(range(self.mesh_x.shape[0])):
                for j in range(self.mesh_x.shape[1]):
                    now_x = self.mesh_x[i, j]
                    now_y = self.mesh_y[i, j]
                    loss = self._compute_loss_for_one_coordinate(now_x, now_y)
                    result.append(loss)

        result = np.array(result)
        result = result.reshape(self.mesh_x.shape)
        return result

    @torch.no_grad()
    def _compute_loss_for_one_coordinate(self, now_x: float, now_y: float):
        temp_model = copy.deepcopy(self.model)
        for name, param in temp_model.named_parameters():
            param.data = param.data + now_x * self.x_unit_vector[name] + now_y * self.y_unit_vector[name]

        return self.loss(temp_model)

    def draw_figure(self, mesh_x, mesh_y, mesh_z):
        if self.mode == '3D':
            figure = plt.figure()
            axes = Axes3D(figure)
            axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')
        elif self.mode == '2D':
            plt.plot(mesh_x, mesh_z)
        plt.savefig(os.path.join(self.save_path, get_datetime_str() + ".png"))
        plt.close()


if __name__ == '__main__':
    pass
