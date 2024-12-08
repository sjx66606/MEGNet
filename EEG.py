# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 21:36
# @Author  : sjx_alo！！
# @FileName: EEG.py
# @Algorithm ：
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

def block1(depth, F1, D, kernel, num_electrodes, dropout=0.25):
    block = nn.Sequential(
        nn.BatchNorm2d(depth),
        nn.Conv2d(depth, F1, (1, kernel), stride=1, padding=(0, kernel // 2), bias=False),
        nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
        Conv2dWithConstraint(F1,
                             F1* D, (num_electrodes, 1),
                             max_norm=1,
                             stride=1,
                             padding=(0, 0),
                             groups=F1,
                             bias=False), nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
        nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))
    return block

def block2(F1, D, F2, kernel, num_electrodes, dropout=0.25):
    block = nn.Sequential(
        nn.Conv2d(F1 * D,
                  F1 * D, (1, kernel),
                  stride=1,
                  padding=(0, kernel // 2),
                  bias=False,
                  groups=F1 * D),
        nn.Conv2d(F1 * D, F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
        nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
        nn.Dropout(p=dropout))
    return block


class EegNet(nn.Module):
    def __init__(self,
                 chunk_size: int = 1125,
                 num_electrodes: int = 22,
                 in_depth = 1,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 4,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 ):
        super(EegNet, self).__init__()

        self.F1 = F1
        self.F2 = F2
        self.in_depth = in_depth
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        # self.block1 = nn.Sequential(
        #     nn.Conv2d(in_depth, 4, (1, 4), stride=(1, 2), bias=False),
        #     nn.BatchNorm2d(4, momentum=0.01, affine=True, eps=1e-3),
        #     # nn.BatchNorm2d(in_depth),
        #     nn.Conv2d(4, self.F1, (1, 125), stride=1, padding=(0, 125 // 2), bias=False),
        #     nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        #     Conv2dWithConstraint(self.F1,
        #                          self.F1 * self.D, (self.num_electrodes, 1),
        #                          max_norm=1,
        #                          stride=1,
        #                          padding=(0, 0),
        #                          groups=self.F1,
        #                          bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
        #     nn.ELU(), nn.AvgPool2d((1, 16), stride=1), nn.Dropout(p=dropout))

        # self.block1_eeg = block1(1, self.F1, self.D, self.kernel_1, self.num_electrodes, self.dropout)
        self.block1_emg = block1(1, self.F1, self.D, self.kernel_1, 4, self.dropout)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_depth, 4, (1, 4), stride=(1, 2), bias=False),
            nn.BatchNorm2d(4, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(4, self.F1, (1, self.kernel_1), stride=1, bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 8), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, 1),
                      stride=1,
                      # padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Sequential(nn.Linear(self.feature_dim, 256, bias=False),
                                 nn.ELU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 32),
                                 nn.ELU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(32, 4))
        # eeg + emg
        # self.lin = nn.Sequential(nn.Linear(2496, num_classes, bias=False),
        #                          nn.Softmax(dim=-1))
        # eeg
        # self.lin = nn.Sequential(nn.Linear(1248, num_classes, bias=False),
        #                          nn.Softmax(dim=-1))


    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)
        return mock_eeg.shape[-1]



    def forward(self, eeg):

        # emg = torch.unsqueeze(emg, dim=1)
        # emgx = self.block1_emg(emg)
        # emgx = self.block2(emgx)
        # emgx = emgx.flatten(start_dim=1)
        # cmcx = self.block1(cmc)
        # cmcx = self.block2(cmcx)
        # cmcx = cmcx.flatten(start_dim=1)
        # eeg = torch.unsqueeze(eeg, dim=1)
        eegx = self.block1(eeg)
        eegx = self.block2(eegx)
        eegx = eegx.flatten(start_dim=1)


        # x = torch.concat([cmcx, eegx, emgx], axis=-1)
        x = eegx
        x = self.lin(x)

        return x.squeeze(-1)