class WarmupLRScheduler:

    def __init__(self, total_epochs, decay_epochs):
        self.warmup_epochs = total_epochs - decay_epochs
        self.decay_epochs = decay_epochs

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return 1.0
        else:
            return 1.0 - ((epoch - self.warmup_epochs) / self.decay_epochs)
