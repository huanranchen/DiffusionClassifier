import torch
from .core import Smooth
from models import resnet50
from torchvision import transforms


class RandomizedSmoothing(torch.nn.Module):
    def __init__(self, sigma="0.50",
                 pretrained=True,
                 ):
        super(RandomizedSmoothing, self).__init__()
        self.device = torch.device('cuda')
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = resnet50()
        if pretrained:
            r = {}
            state = torch.load(
                f'./resources/checkpoints/RS/models/'
                f'imagenet/resnet50/noise_{sigma}/checkpoint.pth.tar')['state_dict']
            for x in list(state.items()):
                if len(x) == 2:
                    k, v = x
                    r[k[9:]] = v
            model.load_state_dict(r)
        model.eval().to(self.device)
        self.model = Smooth(model, 1000, float(sigma))

    def __call__(self, xs):
        xs = self.transform(xs)
        results = []
        for x in xs.split(1, dim=0):
            x = x.squeeze()
            x = self.model.predict(x, n=100, alpha=0.001, batch_size=10)
            result = torch.zeros((1, 1000), device=self.device)
            result[0, x] = 1
            results.append(result)
        results = torch.cat(results, dim=0)
        return results


def randomized_smoothing_resnet50(sigma="0.00", pretrained=True):
    return RandomizedSmoothing(sigma, pretrained=pretrained)
