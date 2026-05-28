import tqdm
from mmcv.runner import Hook, HOOKS


@HOOKS.register_module()
class TqdmHook(Hook):
    """tqdm progress bar for training."""

    def __init__(self, interval=1):
        self.interval = interval

    def before_epoch(self, runner):
        desc = f'Epoch [{runner.epoch + 1}/{runner.max_epochs}]'
        self._pbar = tqdm.tqdm(total=len(runner.data_loader), unit='iter', desc=desc, leave=True, dynamic_ncols=True)

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            lr = runner.current_lr()[0] if runner.current_lr() else 0
            log_vars = runner.outputs.get('log_vars', {})
            loss = log_vars.get('loss', 0)
            loss_ce = log_vars.get('decode.loss_ce', 0)
            acc = log_vars.get('decode.acc_seg', 0)
            self._pbar.set_postfix(loss=f'{loss:.4f}', ce=f'{loss_ce:.4f}', acc=f'{acc:.1f}', lr=f'{lr:.2e}', refresh=False)
            self._pbar.update(self.interval)

    def after_epoch(self, runner):
        self._pbar.close()
