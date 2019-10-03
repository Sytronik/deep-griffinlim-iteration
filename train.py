from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional, Any
from collections import defaultdict

import torch
import numpy as np
from numpy import ndarray
import scipy.io as scio
from torch import nn, Tensor
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from hparams import hp
from dataset import ComplexSpecDataset
# noinspection PyUnresolvedReferences
from model import DeGLI
from tbXwriter import CustomWriter
from utils import arr2str, print_to_file, AverageMeter
from audio_utils import draw_spectrogram


class Trainer:
    def __init__(self, path_state_dict=''):
        self.model = DeGLI(**hp.model)
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

        # Load State Dict
        if path_state_dict:
            st_model, st_optim, st_sched = torch.load(path_state_dict, map_location=self.in_device)
            try:
                if hasattr(self.model, 'module'):
                    self.model.module.load_state_dict(st_model)
                else:
                    self.model.load_state_dict(st_model)
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
            # dd.io.save((hp.logdir / hp.hparams_fname).with_suffix('.h5'), asdict(hp))
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

    def calc_loss(self, outputs: Sequence[Tensor], y: Tensor, T_ys: Sequence[int]) -> Tensor:
        loss = torch.zeros(len(outputs), device=y.device)
        for i, output in enumerate(outputs):
            loss_batch = self.criterion(output, y)
            for T, loss_sample in zip(T_ys, loss_batch):
                loss[i] += torch.mean(loss_sample[:, :, :T])

        weight = torch.tensor([1./i for i in range(len(outputs), 0, -1)], device=y.device)
        loss = loss @ weight / weight.sum()
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
                output_loss, _, _ = self.model(x, mag, max_length)  # B, C, F, T

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
            loss_valid = self.validate(loader_valid, logdir, epoch)
            # self.validate(loader_valid, logdir, epoch, depth=hp.depth_test)

            # save loss & model
            if epoch % hp.period_save_state == hp.period_save_state - 1:
                module = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(
                    (module.state_dict(),
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
    def validate(self, loader: DataLoader, logdir: Path, epoch: int, depth=0):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """

        self.model.eval()

        avg_loss = AverageMeter(float)

        pbar = tqdm(loader, desc='validate ', postfix='[0]', dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, mag, max_length, y = self.preprocess(data)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            output_loss, output, residual = self.model(x, mag, max_length, depth=depth)

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

                self.writer.write_one(epoch, **one_sample, **out_one,
                                      suffix=str(depth) if depth > 0 else '')

        self.writer.add_scalar('loss/valid' + (f'_{depth}' if depth > 0 else ''),
                               avg_loss.get_average(), epoch)

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
            if isinstance(self.model, nn.DataParallel):
                module = self.model.module
            else:
                module = self.model
            for sub in module.children():
                if isinstance(sub, nn.ModuleList):
                    for m in sub:
                        m.register_forward_hook(save_forward)
                elif isinstance(sub, nn.ModuleDict):
                    for m in sub.values():
                        m.register_forward_hook(save_forward)
                else:
                    sub.register_forward_hook(save_forward)

        pbar = tqdm(loader, desc=group, dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, mag, max_length, y = self.preprocess(data)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            if module_counts is not None:
                module_counts = defaultdict(int)

            if i_iter == hp.n_save_block_outs:
                break
            _, output, residual = self.model(x, mag, max_length, depth=hp.depth_test)

            # write summary
            one_sample = ComplexSpecDataset.decollate_padded(data, 0)  # F, T, C

            out_one = self.postprocess(output, residual, T_ys, 0, loader.dataset)

            ComplexSpecDataset.save_dirspec(
                logdir / hp.form_result.format(i_iter),
                **one_sample, **out_one
            )

            measure = self.writer.write_one(i_iter, **out_one, **one_sample,
                                            suffix=str(hp.depth_test))
            if avg_measure is None:
                avg_measure = AverageMeter(init_value=measure, init_count=len(T_ys))
            else:
                avg_measure.update(measure)
            # print
            # str_measure = arr2str(measure).replace('\n', '; ')
            # pbar.write(str_measure)

        self.model.train()

        avg_measure = avg_measure.get_average()

        self.writer.add_text(f'{group}/Average Measure/Proposed', str(avg_measure[0]))
        self.writer.add_text(f'{group}/Average Measure/Reverberant', str(avg_measure[1]))
        self.writer.close()  # Explicitly close

        print()
        str_avg_measure = arr2str(avg_measure).replace('\n', '; ')
        print(f'Average: {str_avg_measure}')
