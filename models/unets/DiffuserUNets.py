from diffusers import DDPMPipeline, DDPMScheduler
from torch import nn


class DDPMCIFARUnet(nn.Module):
    def __init__(self, model_id="google/ddpm-cifar10-32"):
        super(DDPMCIFARUnet, self).__init__()
        # load model and scheduler
        ddpm = DDPMPipeline.from_pretrained(
            model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
        print(ddpm.scheduler)
        self.unet = ddpm.unet

    def forward(self, *args, **kwargs):
        x = self.unet(*args, **kwargs).sample
        return x


