import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO  # BytesIO write bytes in memory

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()

__all__ = ['JPEGCompression']


class Jpeg_compression(object):
    '''
    https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/defense_torch/jpeg_compression.py
    '''

    def __init__(self, quality=75):
        self.quality = quality

    def jpegcompression(self, x):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=self.quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))

    def __call__(self, *args, **kwargs):
        return self.jpegcompression(*args, **kwargs)


class JPEGCompression(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,
                 transform=transforms.Compose([
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])):
        super(JPEGCompression, self).__init__()
        self.model = model
        self.transforms = transform
        self.jpegcompression = Jpeg_compression()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.jpegcompression(x)
        x = x.to(self.device)
        return self.model(x)
