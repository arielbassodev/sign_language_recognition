from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR

class LinearSchedulerWithWarmup(LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            n_warmup_steps: int = 20,
            n_drop_steps: int = 80,
            max_lr: float = 0.002,
            start_lr: float = 0.0,
            end_lr: float = 0.0,
            last_epoch=-1,
    ):
        """
        Linear scheduler with warmup.
        The learning rate increases up to a maximum value during the warm-up, then it decreases.

        Args:
            optimizer: The optimizer on which to schedule the learning rate.
            n_warmup_steps: The number of warmup steps.
            n_drop_steps: The number of steps during which the learning rate drops.
            max_lr: Maximum value of the learning rate (at the end of the warmup).
            start_lr: Starting value of the learning rate (at the beginning of the warmup).
            end_lr: End value of the learning rate (at the end of the drop).
            last_epoch: The index of the last epoch (default=-1).
        """
        self.n_warmup_steps = n_warmup_steps
        self.n_drop_steps = n_drop_steps
        self.max_lr = max_lr
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return [self.start_lr]
        if self.last_epoch < self.n_warmup_steps:
            return [self.start_lr + (self.max_lr - self.start_lr) * self.last_epoch / (self.n_warmup_steps - 1)]
        if self.last_epoch < self.n_warmup_steps + self.n_drop_steps:
            return [self.max_lr - (self.max_lr - self.end_lr) * (self.last_epoch - self.n_warmup_steps) /
                    (self.n_drop_steps - 1)]
        return [self.end_lr]


class WarmupLinearScheduler(LambdaLR):
  def __init__(self, optimizer, warmup_steps=20, t_total=1000, last_epoch=-1):
      self.warmup_steps = warmup_steps
      self.t_total = t_total
      super(WarmupLinearScheduler, self).__init__(
          optimizer, self.lr_lambda, last_epoch=last_epoch
      )
  def lr_lambda(self, step):
      if step < self.warmup_steps:
          return float(step) / float(max(1, self.warmup_steps))
      return max(
          0.0,
          float(self.t_total - step)
          / float(max(1.0, self.t_total - self.warmup_steps)),
      )