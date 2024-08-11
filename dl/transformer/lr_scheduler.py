import torch.nn as nn
import torch.optim as optim


class WarmUpLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_scheduler=None, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            if self.base_scheduler:
                return self.base_scheduler.get_lr()
            else:
                lr = [base_lr for base_lr in self.base_lrs]
        return lr


if __name__ == "__main__":
    # 模型定义
    model = nn.Linear(10, 1)

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 学习率调度器
    base_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = WarmUpLR(optimizer, warmup_steps=10, base_scheduler=base_scheduler)

    # 训练循环
    for epoch in range(100):
        # 训练代码
        # ...

        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch + 1}, LR: {scheduler.get_lr()}")
