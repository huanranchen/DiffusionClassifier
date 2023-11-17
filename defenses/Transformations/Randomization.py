import torch
import torch.nn.functional as F
from torchvision import transforms
import random

__all__ = ['Randomization']


class RandomizationFunction(object):
    '''
    reference:
    https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/defense_torch/randomization.py
    '''

    def __init__(self, prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        self.prob = prob
        self.crop_lst = crop_lst

    def input_transform(self, xs):
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs):
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

        if len(xs.shape) == 4:
            bs, c, w, h = xs.shape
        elif len(xs.shape) == 5:
            bs, fs, c, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)
        # out = resize(xs, size=(w_, h_))
        out = F.interpolate(xs, size=[w_, h_], mode='bicubic', align_corners=False)

        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        out = F.pad(out, [pad_left, w - pad_left - w_, pad_top, h - pad_top - h_], value=0)
        return out

    def __call__(self, *args, **kwargs):
        return self.input_transform(*args, **kwargs)


class Randomization(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,
                 transform=transforms.Compose([
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])):
        super(Randomization, self).__init__()
        self.model = model
        self.transforms = transform
        self.randomization = RandomizationFunction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.i = 0

    def forward(self, x):
        x = self.randomization(x)
        return self.model(x)

# class Randomization(torch.nn.Module):
#     def __init__(self, model: torch.nn.Module,
#                  transform=transforms.Compose([
#                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                  ])):
#         super(Randomization, self).__init__()
#         self.model = model
#         self.transforms = transform
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.i=0
#
#     def forward(self, x):
#         # random resize
#         resized_shape = random.randint(299, 331)
#         resizer = transforms.Resize((resized_shape, resized_shape))
#         x = resizer(x)
#         # random pad
#         pad_size = 331 - resized_shape
#         left_pad_size = random.randint(0, pad_size)
#         up_pad_size = random.randint(0, pad_size)
#         x = F.pad(x, (left_pad_size, pad_size - left_pad_size, up_pad_size, pad_size - up_pad_size))
#         img = transforms.ToPILImage()(x[0])
#         img.save(f'./what/{self.i}.png')
#         self.i += 1
#         x = self.transforms(x)
#         return self.model(x)
