# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np
import random

from .common import ResNet18, MLP


class Net(torch.nn.Module):
    # Re-implementation of Learning with Forgetting
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.nt = n_tasks
        self.reg = args.memory_strength
        self.n_feat = n_outputs
        self.n_classes = n_outputs
        self.samples_per_task = args.samples_per_task
        if self.samples_per_task <= 0:
            error("set explicitly args.samples_per_task")
        self.examples_seen = 0

        # setup network
        # assert (args.data_file == 'cifar100.pt')
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
            self.old_net = ResNet18(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            self.old_net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss()
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        self.gpu = args.cuda
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

    def compute_offsets(self, task):
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.net.train()
        self.net.zero_grad()
        self.old_net.eval()

        self.examples_seen += x.size(0)

        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            logits = self.net(x)
            # loss = self.bce(logits[:, offset1:offset2], y - offset1)
            loss = self.bce(logits[:, :offset2], y)

            if t > 0:
                # Add distillation loss
                old_logits = self.old_net(x).clone()
                for tt in range(t):
                    offset1, offset2 = self.compute_offsets(tt)
                    loss += self.reg * self.kl(
                        self.lsm(logits[:, offset1:offset2]),
                        self.sm(
                            old_logits[:, offset1:offset2])) * self.nc_per_task
        else:
            logits = self(x, t)
            loss = self.bce(logits, y)
            if t > 0:
                # Add distillation loss
                old_logits = self.old_net(x).clone()
                for tt in range(t):
                    loss += self.reg * self.kl(
                        self.lsm(logits),
                        self.sm(old_logits)) * self.nc_per_task

        # bprop and update
        loss.backward()
        self.opt.step()
