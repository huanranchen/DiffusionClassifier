import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from .ColorUtils import get_rand_cmap, suppress_stdout_stderr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

modes = ['3D', 'Contour', 'HeatMap', '2D']
alpha = 0.5


class Landscape4Input():
    def __init__(self, model,
                 input: torch.tensor,
                 mode='3D'):
        '''

        :param model: taken input as input, output loss
        :param input:
        '''
        self.model = model
        self.input = input
        assert mode in modes
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def synthesize_coordinates(self,
                               x_min=-16 / 255, x_max=16 / 255, x_interval=1 / 255,
                               y_min=-16 / 255, y_max=16 / 255, y_interval=1 / 255):
        x = np.arange(x_min, x_max, x_interval)
        y = np.arange(y_min, y_max, y_interval)
        self.mesh_x, self.mesh_y = np.meshgrid(x, y)
        return self.mesh_x, self.mesh_y

    def assign_coordinates(self, x, y):
        self.mesh_x = x
        self.mesh_y = y

    def assign_unit_vector(self, x_unit_vector, y_unit_vector=None):
        self.x_unit_vector = x_unit_vector
        self.y_unit_vector = y_unit_vector

    @torch.no_grad()
    def draw(self, axes=None):
        if hasattr(self, 'x_unit_vector') and self.x_unit_vector is not None:
            pass
        else:
            self._find_direction()
        z = self._compute_for_draw()
        if axes is None and self.mode == '3D':
            axes = plt.axes(projection='3d')
        self._draw3D(self.mesh_x, self.mesh_y, z, axes)

    def _find_direction(self):
        self.x_unit_vector = torch.randn(self.input.shape, device=self.device)
        self.y_unit_vector = torch.randn(self.input.shape, device=self.device)
        self.x_unit_vector /= torch.norm(self.x_unit_vector, p=float('inf'))
        self.y_unit_vector /= torch.norm(self.y_unit_vector, p=float('inf'))  # make sure the l 2 norm is 0
        # keep perpendicular
        # if torch.abs(self.x0.reshape(-1) @ self.y0.reshape(-1)) >= 0.1:
        #     self._find_direction()

    def _compute_for_draw(self):
        result = []
        if self.mode == '2D':
            self.mesh_x = self.mesh_x[0, :]
            for i in tqdm(range(self.mesh_x.shape[0])):
                # with suppress_stdout_stderr():
                    now_x = self.mesh_x[i]
                    x = self.input + self.x_unit_vector * now_x
                    x = self.project(x)
                    loss = self.model(x)
                    result.append(loss)
        else:
            for i in tqdm(range(self.mesh_x.shape[0])):
                # with suppress_stdout_stderr():
                    for j in range(self.mesh_x.shape[1]):
                        now_x = self.mesh_x[i, j]
                        now_y = self.mesh_y[i, j]
                        x = self.input + self.x_unit_vector * now_x + self.y_unit_vector * now_y
                        x = self.project(x)
                        loss = self.model(x)
                        result.append(loss)
        result = np.array(result)
        result = result.reshape(self.mesh_x.shape)
        return result

    def _draw3D(self, mesh_x, mesh_y, mesh_z, axes=None):
        if self.mode == '3D':
            axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')

        if self.mode == 'Contour':
            plt.contourf(mesh_x, mesh_y, mesh_z, 1, cmap=get_rand_cmap(), alpha=alpha)

        if self.mode == '2D':
            plt.plot(mesh_x, mesh_z)

        # plt.show()
        # plt.close()
        # plt.savefig(self.get_datetime_str() + ".png")

    @staticmethod
    def get_datetime_str(style='dt'):
        import datetime
        cur_time = datetime.datetime.now()

        date_str = cur_time.strftime('%y_%m_%d_')
        time_str = cur_time.strftime('%H_%M_%S')

        if style == 'data':
            return date_str
        elif style == 'time':
            return time_str
        else:
            return date_str + time_str

    @staticmethod
    def project(x: torch.tensor, min=0, max=1):
        return torch.clamp(x, min, max)
