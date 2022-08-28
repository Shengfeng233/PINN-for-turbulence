# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
# import torch
# import torch.optim as optim
# from torchvision.models import AlexNet
# from torch.optim import lr_scheduler
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# model = AlexNet(num_classes=2)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-1)
# scheduler = ChainedScheduler(
#     optimizer,
#     T_0 = 50,
#     T_mul = 2,
#     eta_min = 0.0,
#     gamma = 0.9,
#     max_lr = 0.001,
#     warmup_steps=0,
# )
#
#
# fig = matplotlib.pyplot.gcf()
# matplotlib.rcParams['figure.figsize'] = [18.5, 10]
# x = list(range(3000))
# y = []
#
# for epoch in range(3000):
#     optimizer.step()
#     scheduler.step()
#     y.append(scheduler.get_lr()[0])
#
# fig, axes = plt.subplots(1, 1)
# xticks = range(min(x), max(x) + 1)
# y_mat = np.array(y).reshape(-1, 1)
# # axes.set_xticks(xticks)
# plt.plot(xticks, y)
# # plt.grid()
# plt.show()
import math
import torch
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    """
    Args:
        optimizer: [torch.optim.Optimizer] only pass if using as astand alone lr_scheduler
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            eta_min: float = 0.0,
            last_epoch=-1,
            max_lr: Optional[float] = 0.1,
            warmup_steps: Optional[int] = 0,
    ):

        if warmup_steps != 0:
            assert warmup_steps >= 0

        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.step_in_cycle = last_epoch
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup

        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]

        else:
            return [base_lr + (self.max_lr - base_lr) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

        else:
            self.step_in_cycle = self.epoch

        self.max_lr = self.base_max_lr
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnealingWarmRestartsWeightDecay(_LRScheduler):
    """
       Helper class for chained scheduler not to used directly. this class is synchronised with
       previous stage i.e.  WarmUpScheduler (max_lr, T_0, T_cur etc) and is responsible for
       CosineAnealingWarmRestarts with weight decay
       """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_0: int,
            T_mul: float = 1.,
            eta_min: float = 0.001,
            last_epoch=-1,
            max_lr: Optional[float] = 0.1,
            gamma: Optional[float] = 1.,
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        super(CosineAnealingWarmRestartsWeightDecay, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mul

        # since warmup steps must be < T_0 and if epoch count > T_0 we just apply cycle count for weight decay
        if self.epoch >= self.T_0:
            if self.T_mul == 1.:
                self.T_cur = self.epoch % self.T_0
                self.cycle = self.epoch // self.T_0
            else:
                n = int(math.log((self.epoch / self.T_0 * (self.T_mul - 1) + 1), self.T_mul))
                self.cycle = n
                self.T_cur = self.epoch - int(self.T_0 * (self.T_mul ** n - 1) / (self.T_mul - 1))
                self.T_i = self.T_0 * self.T_mul ** (n)

        # base condition that applies original implementation for cosine cycles for details visit:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        else:
            self.T_i = self.T_0
            self.T_cur = self.epoch

        # this is where weight decay is applied
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ChainedScheduler(_LRScheduler):
    """
    Driver class
        Args:
        T_0: First cycle step size, Number of iterations for the first restart.
        T_mul: multiplicative factor Default: -1., A factor increases T_i after a restart
        eta_min: Min learning rate. Default: 0.001.
        max_lr: warmup's max learning rate. Default: 0.1. shared between both schedulers
        warmup_steps: Linear warmup step size. Number of iterations to complete the warmup
        gamma: Decrease rate of max learning rate by cycle. Default: 1.0 i.e. no decay
        last_epoch: The index of last epoch. Default: -1
    Usage:
        ChainedScheduler without initial warmup and weight decay:
            scheduler = ChainedScheduler(
                            optimizer,
                            T_0=20,
                            T_mul=2,
                            eta_min = 1e-5,
                            warmup_steps=0,
                            gamma = 1.0
                        )
        ChainedScheduler with weight decay only:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch=-1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 0,
                            gamma: Optional[float] = 0.9
                        )
        ChainedScheduler with initial warm up and weight decay:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch = -1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 10,
                            gamma: Optional[float] = 0.9
                        )
    Example:
        >>> model = AlexNet(num_classes=2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-1)
        >>> scheduler = ChainedScheduler(
        >>>                 optimizer,
        >>>                 T_0 = 20,
        >>>                 T_mul = 1,
        >>>                 eta_min = 0.0,
        >>>                 gamma = 0.9,
        >>>                 max_lr = 1.0,
        >>>                 warmup_steps= 5 ,
        >>>             )
        >>> for epoch in range(100):
        >>>     optimizer.step()
        >>>     scheduler.step()
    Proper Usage:
        https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_0: int,
            T_mul: float = 1.0,
            eta_min: float = 0.001,
            last_epoch=-1,
            max_lr: Optional[float] = 1.0,
            warmup_steps: Optional[int] = 5,
            gamma: Optional[float] = 0.95,
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        if warmup_steps != 0:
            assert warmup_steps < T_0
            warmup_steps = warmup_steps + 1  # directly refers to epoch account for 0 off set

        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        self.last_epoch = last_epoch

        self.cosine_scheduler1 = WarmUpScheduler(
            optimizer,
            eta_min=self.eta_min,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
        )
        self.cosine_scheduler2 = CosineAnealingWarmRestartsWeightDecay(
            optimizer,
            T_0=self.T_0,
            T_mul=self.T_mul,
            eta_min=self.eta_min,
            max_lr=self.max_lr,
            gamma=self.gamma,
        )

    def get_lr(self):
        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                return self.cosine_scheduler1.get_lr()
        if self.epoch >= self.warmup_steps:
            return self.cosine_scheduler2.get_lr()

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1

        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                self.cosine_scheduler1.step()
                self.last_epoch = self.epoch

        if self.epoch >= self.warmup_steps:
            self.cosine_scheduler2.step()
            self.last_epoch = self.epoch

# import torch
# import torch.optim as optim
# from torchvision.models import AlexNet
# from torch.optim import lr_scheduler
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# model = AlexNet(num_classes=2)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-1)
# scheduler = ChainedScheduler(
#     optimizer,
#     T_0 = 50,
#     T_mul = 2,
#     eta_min = 0.0,
#     gamma = 0.9,
#     max_lr = 0.001,
#     warmup_steps=0,
# )
#
#
# fig = matplotlib.pyplot.gcf()
# matplotlib.rcParams['figure.figsize'] = [18.5, 10]
# x = list(range(3000))
# y = []
#
# for epoch in range(3000):
#     optimizer.step()
#     scheduler.step()
#     y.append(scheduler.get_lr()[0])
#
# fig, axes = plt.subplots(1, 1)
# xticks = range(min(x), max(x) + 1)
# y_mat = np.array(y).reshape(-1, 1)
# # axes.set_xticks(xticks)
# plt.plot(xticks, y)
# # plt.grid()
# plt.show()