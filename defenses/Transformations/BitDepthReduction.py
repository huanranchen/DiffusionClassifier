import torch
from torchvision import transforms

__all__ = ['BitDepthReduction']

class BitDepthReductionFunction(object):
    '''
    https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/defense_torch/bit_depth_reduction.py
    '''

    def __init__(self, compressed_bit=4):
        self.compressed_bit = compressed_bit
        self.device = torch.device('cuda')

    def bit_depth_reduction(self, xs):
        bits = 2 ** self.compressed_bit  # 2**i
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = (xs_255 / 255).to(self.device)
        return xs_compress

    def __call__(self, *args, **kwargs):
        return self.bit_depth_reduction(*args, **kwargs)


class BitDepthReduction(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,):
        super(BitDepthReduction, self).__init__()
        self.model = model
        self.bit = BitDepthReductionFunction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.bit(x)
        return self.model(x)
