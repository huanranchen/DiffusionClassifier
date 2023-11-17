class IdentityScheduler:
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class ALRS:
    """
    refer to
    Bootstrap Generalization Ability from Loss Landscape Perspective
    """

    def __init__(self, optimizer, loss_threshold=0.01, loss_ratio_threshold=0.01, decay_rate=0.97):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.decay_rate
                now_lr = group["lr"]
                print(f"now lr = {now_lr}")

        self.last_loss = loss
