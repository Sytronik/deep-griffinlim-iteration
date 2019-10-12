import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.optim.lr_scheduler as lr_scheduler
from numpy import ndarray
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torchsummary import summary
from tqdm import tqdm

from dataset import ComplexSpecDataset
from hparams import hp
# noinspection PyUnresolvedReferences
from model import DeGLI
from tbwriter import CustomWriter
from utils import AverageMeter, arr2str, draw_spectrogram, print_to_file


class Trainer:
    def __init__(self, path_state_dict=''):
        self.model = DeGLI(**hp.model)
        self.module = self.model
        self.criterion = nn.L1Loss(reduction='none')
        self.optimizer = Adam(self.model.parameters(),
                              lr=hp.learning_rate,
                              weight_decay=hp.weight_decay,
                              )

        self.__init_device(hp.device, hp.out_device)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **hp.scheduler)
        self.max_epochs = hp.n_epochs

        self.writer: Optional[CustomWriter] = None

        self.valid_eval_sample: Dict[str, Any] = dict()

        # if hp.model['final_avg']:
        #     len_weight = hp.repeat_train
        # else:
        #     len_weight = hp.model['depth'] * hp.repeat_train
        len_weight = hp.repeat_train
        self.loss_weight = torch.tensor(
            [1./i for i in range(len_weight, 0, -1)],
            device=self.out_device,
        )
        self.loss_weight /= self.loss_weight.sum()

        # Load State Dict
        if path_state_dict:
            st_model, st_optim, st_sched = torch.load(path_state_dict, map_location=self.in_device)
            try:
                self.module.load_state_dict(st_model)
                self.optimizer.load_state_dict(st_optim)
                self.scheduler.load_state_dict(st_sched)
            except:
                raise Exception('The model is different from the state dict.')

        path_summary = hp.logdir / 'summary.txt'
        if not path_summary.exists():
            # print_to_file(
            #     path_summary,
            #     summary,
            #     (self.model, hp.dummy_input_size),
            #     dict(device=self.str_device[:4])
            # )
            with path_summary.open('w') as f:
                f.write('\n')
            with (hp.logdir / 'hparams.txt').open('w') as f:
                f.write(repr(hp))

    def __init_device(self, device, out_device):
        """

        :type device: Union[int, str, Sequence]
        :type out_device: Union[int, str, Sequence]
        :return:
        """
        if device == 'cpu':
            self.in_device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return

        # device type: List[int]
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device.replace('cuda:', ''))]
        else:  # sequence of devices
            if type(device[0]) != int:
                device = [int(d.replace('cuda:', '')) for d in device]

        self.in_device = torch.device(f'cuda:{device[0]}')

        if len(device) > 1:
            if type(out_device) == int:
                self.out_device = torch.device(f'cuda:{out_device}')
            else:
                self.out_device = torch.device(out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])

            self.model = nn.DataParallel(self.model,
                                         device_ids=device,
                                         output_device=self.out_device)
        else:
            self.out_device = self.in_device
            self.str_device = str(self.in_device)

        self.model.cuda(self.in_device)
        self.criterion.cuda(self.out_device)

        torch.cuda.set_device(self.in_device)

    def preprocess(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        x = data['x']
        mag = data['y_mag']
        max_length = max(data['length'])
        y = data['y']

        x = x.to(self.in_device, non_blocking=True)
        mag = mag.to(self.in_device, non_blocking=True)
        y = y.to(self.out_device, non_blocking=True)

        return x, mag, max_length, y

    @torch.no_grad()
    def postprocess(self, output: Tensor, residual: Tensor, Ts: ndarray, idx: int,
                    dataset: ComplexSpecDataset) -> Dict[str, ndarray]:
        dict_one = dict(out=output, res=residual)
        for key in dict_one:
            one = dict_one[key][idx, :, :, :Ts[idx]]
            one = one.permute(1, 2, 0).contiguous()  # F, T, 2

            one = one.cpu().numpy().view(dtype=np.complex64)  # F, T, 1
            dict_one[key] = one

        return dict_one

    def calc_loss(self, out_blocks: Tensor, y: Tensor, T_ys: Sequence[int]) -> Tensor:
        """
        out_blocks: B, depth, C, F, T
        y: B, C, F, T
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = self.criterion(out_blocks, y.unsqueeze(1))
        loss_blocks = torch.zeros(out_blocks.shape[1], device=y.device)
        for T, loss_batch in zip(T_ys, loss_no_red):
            loss_blocks += torch.mean(loss_batch[..., :T], dim=(1, 2, 3))

        if len(loss_blocks) == 1:
            loss = loss_blocks.squeeze()
        else:
            loss = loss_blocks @ self.loss_weight
        return loss

    @torch.no_grad()
    def should_stop(self, loss_valid, epoch):
        if epoch == self.max_epochs - 1:
            return True
        self.scheduler.step(loss_valid)
        # if self.scheduler.t_epoch == 0:  # if it is restarted now
        #     # if self.loss_last_restart < loss_valid:
        #     #     return True
        #     if self.loss_last_restart * hp.threshold_stop < loss_valid:
        #         self.max_epochs = epoch + self.scheduler.restart_period + 1
        #     self.loss_last_restart = loss_valid

    def train(self, loader_train: DataLoader, loader_valid: DataLoader,
              logdir: Path, first_epoch=0):
        self.writer = CustomWriter(str(logdir), group='train', purge_step=first_epoch)

        # Start Training
        for epoch in range(first_epoch, hp.n_epochs):
            self.writer.add_scalar('loss/lr', self.optimizer.param_groups[0]['lr'], epoch)
            print()
            pbar = tqdm(loader_train,
                        desc=f'epoch {epoch:3d}', postfix='[]', dynamic_ncols=True)
            avg_loss = AverageMeter(float)
            avg_grad_norm = AverageMeter(float)

            for i_iter, data in enumerate(pbar):
                # get data
                x, mag, max_length, y = self.preprocess(data)  # B, C, F, T
                T_ys = data['T_ys']

                # forward
                output_loss, _, _ = self.model(x, mag, max_length,
                                               repeat=hp.repeat_train)  # B, C, F, T

                loss = self.calc_loss(output_loss, y, T_ys)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           hp.thr_clip_grad)

                self.optimizer.step()

                # print
                avg_loss.update(loss.item(), len(T_ys))
                pbar.set_postfix_str(f'{avg_loss.get_average():.1e}')
                avg_grad_norm.update(grad_norm)

            self.writer.add_scalar('loss/train', avg_loss.get_average(), epoch)
            self.writer.add_scalar('loss/grad', avg_grad_norm.get_average(), epoch)

            # Validation
            # loss_valid = self.validate(loader_valid, logdir, epoch)
            loss_valid = self.validate(loader_valid, logdir, epoch, repeat=hp.repeat_train)

            # save loss & model
            if epoch % hp.period_save_state == hp.period_save_state - 1:
                torch.save(
                    (self.module.state_dict(),
                     self.optimizer.state_dict(),
                     self.scheduler.state_dict(),
                     ),
                    logdir / f'{epoch}.pt'
                )

            # Early stopping
            if self.should_stop(loss_valid, epoch):
                break

        self.writer.close()

    @torch.no_grad()
    def validate(self, loader: DataLoader, logdir: Path, epoch: int, repeat=1):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """
        suffix = f'_{repeat}' if repeat > 1 else ''

        self.model.eval()

        avg_loss = AverageMeter(float)

        pbar = tqdm(loader, desc='validate ', postfix='[0]', dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, mag, max_length, y = self.preprocess(data)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            output_loss, output, residual = self.model(x, mag, max_length,
                                                       repeat=repeat)

            # loss
            loss = self.calc_loss(output_loss, y, T_ys)
            avg_loss.update(loss.item(), len(T_ys))

            # print
            pbar.set_postfix_str(f'{avg_loss.get_average():.1e}')

            # write summary
            if i_iter == 0:
                # F, T, C
                if not self.valid_eval_sample:
                    self.valid_eval_sample = ComplexSpecDataset.decollate_padded(data, 0)

                out_one = self.postprocess(output, residual, T_ys, 0, loader.dataset)

                # ComplexSpecDataset.save_dirspec(
                #     logdir / hp.form_result.format(epoch),
                #     **self.valid_eval_sample, **out_one
                # )

                if not self.writer.reused_sample:
                    one_sample = self.valid_eval_sample
                else:
                    one_sample = dict()

                self.writer.write_one(epoch, **one_sample, **out_one, suffix=suffix)

        self.writer.add_scalar(f'loss/valid{suffix}', avg_loss.get_average(), epoch)

        self.model.train()

        return avg_loss.get_average()

    @torch.no_grad()
    def test(self, loader: DataLoader, logdir: Path):
        def save_forward(module: nn.Module, in_: Tensor, out: Tensor):
            module_name = str(module).split('(')[0]
            dict_to_save = dict()
            # dict_to_save['in'] = in_.detach().cpu().numpy().squeeze()
            dict_to_save['out'] = out.detach().cpu().numpy().squeeze()

            i_module = module_counts[module_name]
            for i, o in enumerate(dict_to_save['out']):
                save_forward.writer.add_figure(
                    f'{group}/blockout_{i_iter}/{module_name}{i_module}',
                    draw_spectrogram(o, to_db=False),
                    i,
                )
            scio.savemat(
                str(logdir / f'blockout_{i_iter}_{module_name}{i_module}.mat'),
                dict_to_save,
            )
            module_counts[module_name] += 1

        group = logdir.name.split('_')[0]

        self.writer = CustomWriter(str(logdir), group=group)

        avg_measure = None
        self.model.eval()

        module_counts = None
        if hp.n_save_block_outs:
            module_counts = defaultdict(int)
            save_forward.writer = self.writer
            for sub in self.module.children():
                if isinstance(sub, nn.ModuleList):
                    for m in sub:
                        m.register_forward_hook(save_forward)
                elif isinstance(sub, nn.ModuleDict):
                    for m in sub.values():
                        m.register_forward_hook(save_forward)
                else:
                    sub.register_forward_hook(save_forward)

        pbar = tqdm(loader, desc=group, dynamic_ncols=True)
        cnt_sample = 0
        for i_iter, data in enumerate(pbar):
            # get data
            x, mag, max_length, y = self.preprocess(data)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            if module_counts is not None:
                module_counts = defaultdict(int)

            if 0 < hp.n_save_block_outs == i_iter:
                break
            _, output, residual = self.model(x, mag, max_length,
                                             repeat=hp.repeat_test)

            # write summary
            for i_b in range(len(T_ys)):
                i_sample = cnt_sample + i_b
                one_sample = ComplexSpecDataset.decollate_padded(data, i_b)  # F, T, C

                out_one = self.postprocess(output, residual, T_ys, i_b, loader.dataset)

                ComplexSpecDataset.save_dirspec(
                    logdir / hp.form_result.format(i_sample),
                    **one_sample, **out_one
                )

                measure = self.writer.write_one(i_sample, **out_one, **one_sample,
                                                suffix=f'_{hp.repeat_test}')
                if avg_measure is None:
                    avg_measure = AverageMeter(init_value=measure)
                else:
                    avg_measure.update(measure)
                # print
                # str_measure = arr2str(measure).replace('\n', '; ')
                # pbar.write(str_measure)
            cnt_sample += len(T_ys)

        self.model.train()

        avg_measure = avg_measure.get_average()

        self.writer.add_text(f'{group}/Average Measure/Proposed', str(avg_measure[0]))
        self.writer.add_text(f'{group}/Average Measure/Reverberant', str(avg_measure[1]))
        self.writer.close()  # Explicitly close

        print()
        str_avg_measure = arr2str(avg_measure).replace('\n', '; ')
        print(f'Average: {str_avg_measure}')
