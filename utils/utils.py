import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch import optim, nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
def print_sample_with_rare_classes(trg, output, loss, 
                                   i, name):
    path = os.path.join('log', name)
    if torch.any(trg < 3) or torch.any(trg > 5):
        with open(path, 'a') as f:
            print(f'Training loss at i = {i}', file= f)
            print(trg, file= f)
            print(output, file= f)
            print(loss.item(), file= f)

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.0001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.0001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=True, file_name = 'model.pt', dir_='checkpoints'):
        """
        Args:
            patience (int): How long to wait after last improvement before stopping.
                           Default: 10
            delta (float): Minimum change in monitored quantity to qualify as improvement.
                           Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                           Default: False
            path (str): Path to save the checkpoint when the best validation loss is found.
                        Default: 'checkpoint'
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = os.path.join(dir_, file_name)

    def __call__(self, val_loss, model):
        score = -val_loss  # Assuming you are monitoring loss, so higher is worse.

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.val_loss_min < 0.75:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


def train(model: nn.Module,
          train_iterator: torch.utils.data.DataLoader,
          test_iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float,
          evaluate,
          early_stop,
          scheduler = None,
          name = 'model'):


    train_epoch_loss = 0

    for i, (c_datas, d_datas, trg, seq_lengths) in enumerate(train_iterator):
        if early_stop.early_stop:
            break
        trg = trg.to(device)
        c_datas = c_datas.to(device)
        d_datas = d_datas.to(device)
        optimizer.zero_grad()
        output = model(c_datas, d_datas, seq_lengths)
        output = output.squeeze(-1)
        # weights = compute_weights(trg)
        loss = criterion(output, trg)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # scheduler.step()
        optimizer.step()
        train_epoch_loss += loss.item()
        print_sample_with_rare_classes(trg, output, loss, i, name)
        # if i != 0 and i % 3 == 0:
        #     print(f'Training loss at i = {i}')
        #     print(trg)
        #     print(output)
        #     print(loss.item())
        if i != 0 and i % 50 == 0:
            val_rmse_loss = evaluate(model, test_iterator,
                                                criterion)
            early_stop(val_rmse_loss, model)
            print(f'Training loss at i = {i}')
            print(trg)
            print(output)
            print(loss.item() )
            print(f'\t Val. Loss: {val_rmse_loss:.3f}')

    return train_epoch_loss / len(train_iterator)

def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):


    epoch_loss = 0

    with torch.no_grad():

        for _, (c_datas, d_datas, trg, seq_lengths) in enumerate(iterator):
            trg = trg.to(device)
            c_datas = c_datas.to(device)
            d_datas = d_datas.to(device)
            output = model(c_datas, d_datas, seq_lengths)
            output = output.squeeze(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return math.sqrt(epoch_loss / len(iterator))