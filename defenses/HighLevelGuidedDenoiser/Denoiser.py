import torch
from attacks import AdversarialInputAttacker
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torchvision import transforms


class PixelGuidedDenoiser():
    def __init__(self,
                 attacker: AdversarialInputAttacker,
                 unet: torch.nn.Module,
                 device=torch.device('cuda'),
                 ):
        self.student = unet
        self.attacker = attacker
        self.device = device
        self.writer = SummaryWriter(log_dir="runs/Solver", flush_secs=120)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)
        self.to_img = transforms.ToPILImage()

    def save(self, path='./resources/checkpoints/denoiser/pixel_level_denoiser.pth'):
        torch.save(self.student.state_dict(), path)
        print('-' * 100)
        print('managed to save PixelGuidedDenoiser weight')
        print('-' * 100)

    def train(self,
              loader: DataLoader,
              total_epoch=1000,
              fp16=False,
              ):
        '''
        :param total_epoch:
        :param step_each_epoch: this 2 parameters is just a convention, for when output loss and acc, etc.
        :param fp16:
        :param generating_data_configuration:
        :return:
        '''
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        self.student.train()
        for epoch in range(1, total_epoch + 1):
            train_loss = 0
            pbar = tqdm(loader)
            for step, (x, y) in enumerate(pbar, 1):
                self.writer.add_image('image/origin', self.to_img(x[0]))
                x, y = x.to(self.device), y.to(self.device)
                adv_x = self.attacker(x, y)
                y = x
                x = adv_x
                if fp16:
                    with autocast():
                        student_out = self.student(x)  # N, 60
                        loss = self.criterion(student_out, y)
                else:
                    student_out = self.student(x)  # N, 60
                    loss = self.criterion(student_out, y)
                self.writer.add_image('image/origin', self.to_img(student_out[0]))
                train_loss += loss.item()

                self.optimizer.zero_grad()
                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    self.optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={train_loss / step}')

            train_loss /= len(loader)
            print(f'epoch {epoch}, loss = {train_loss}')
            torch.save(self.student.state_dict(), 'student.pth')
            self.writer.add_scalar('scaler/loss', train_loss)



