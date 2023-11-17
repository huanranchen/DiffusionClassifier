import torch
from .networks import NRP, NRP_resG
from torchvision import transforms


class NeuralRepresentationPurifier(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,
                 transform=transforms.Compose([
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]),
                 mode='NRP_resG',
                 ):
        super(NeuralRepresentationPurifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.transforms = transform
        if mode == 'NRP':
            self.netG = NRP(3, 3, 64, 23)
            self.netG.load_state_dict(torch.load('resources/checkpoints/NRP/NRP.pth'))
        if mode == 'NRP_resG':
            self.netG = NRP_resG(3, 3, 64, 23)
            self.netG.load_state_dict(torch.load('resources/checkpoints/NRP/NRP_resG.pth'))
        self.netG.to(device=self.device)
        self.i = 0
        self.to_img = transforms.ToPILImage()

    def forward(self, x):
        x = self.netG(x)
        return self.model(x)

